import sys
import os
import csv
import argparse
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import text
from sqlalchemy.orm import Session

from config.db import get_legacy_engine, get_new_engine
from config.config import BATCH_SIZE, DRY_RUN, PREVIEW_SAMPLE_SIZE
from utils.id_gen import SafeIDGenerator, get_migrated_legacy_ids
from utils.logger import get_logger

log = get_logger("migrate_documents")

OLD_S3_BUCKET      = os.environ.get("OLD_S3_BUCKET", "ds-prod-new")
NEW_S3_BUCKET      = os.environ.get("NEW_S3_BUCKET", "digiaarogyasaarathifiles")
AWS_REGION         = os.environ.get("AWS_REGION", "ap-south-1")
S3_PUBLIC_BASE_URL = os.environ.get(
    "S3_PUBLIC_BASE_URL",
    f"https://{NEW_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com",
).rstrip("/")

CATEGORY   = "PATIENT_MEDICAL_RECORDS"
VISIBILITY = "['doctor', 'coordinator']"

LEGACY_SOURCE_HCA        = "healthcase_attached"
VISIT_LEGACY_SOURCE      = "digiswasthya_database"   # must match migrate_visit.py's LEGACY_SOURCE

DEFAULT_PREDICTIONS_CSV  = os.environ.get("PREDICTIONS_CSV_PATH", "predictions.csv")

# Only these predictions are eligible for migration. Everything else is skipped.
ALLOWED_PREDICTIONS = {
    "Prescription / Document",
    "X-ray / Sonography",
}

# document_type in the new schema, driven by the CSV prediction.
PREDICTION_TO_DOCUMENT_TYPE = {
    "Prescription / Document": "PRESCRIPTION",
    "X-ray / Sonography":      "XRAY",
}


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def copy_s3_object(s3, old_key: str, new_key: str) -> bool:
    if DRY_RUN:
        log.info("[DRY RUN] S3 copy  %s/%s  ->  %s/%s", OLD_S3_BUCKET, old_key, NEW_S3_BUCKET, new_key)
        return True
    try:
        s3.copy_object(
            CopySource={"Bucket": OLD_S3_BUCKET, "Key": old_key},
            Bucket=NEW_S3_BUCKET,
            Key=new_key,
        )
        log.debug("S3 copied  %s  ->  %s", old_key, new_key)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            log.warning("S3 source not found, skipping:  %s/%s", OLD_S3_BUCKET, old_key)
        else:
            log.error("S3 copy failed  %s -> %s: %s", old_key, new_key, e)
        return False
    except Exception as e:
        log.error("S3 copy failed  %s -> %s: %s", old_key, new_key, e)
        return False


# ---------------------------------------------------------------------------
# Key / URL helpers
# ---------------------------------------------------------------------------

def filename_from_key(key_or_url: str) -> str:
    """Bare filename (last path segment), stripped of query params if any."""
    cleaned = (key_or_url or "").strip().split("?")[0]
    return cleaned.split("/")[-1]


def build_new_s3_key(new_patient_id: str, old_key: str) -> str:
    """
    Flatten all images directly under the patient folder, stripping any
    subfolder segments from the old key.

    OLD:  ds-prod-new/2xsqqgw/image_1771653159275_pgn05i.jpg
    NEW:  digiaarogyasaarathifiles/documents/{new_patient_id}/image_1771653159275_pgn05i.jpg
    """
    return f"documents/{new_patient_id}/{filename_from_key(old_key)}"


def build_storage_url(storage_key: str) -> str:
    return f"{S3_PUBLIC_BASE_URL}/{storage_key}"


def content_type_from_key(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    return {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
        "pdf":  "application/pdf",
    }.get(ext, "application/octet-stream")


def source_from_uploaderrole(role: str | None) -> str:
    if not role:
        return "MIGRATION"
    role_lower = (role or "").strip().lower()
    if role_lower == "coordinator":
        return "coordinator_portal"
    if role_lower == "doctor":
        return "doctor_portal"
    return "MIGRATION"


# ---------------------------------------------------------------------------
# Prediction CSV
# ---------------------------------------------------------------------------

def load_prediction_lookup(csv_path: str) -> dict[str, dict]:
    """
    Load the prediction CSV keyed by bare filename (extracted from s3_url).

    Returns: { filename: {"prediction": str, "confidence": float,
                           "xray_score": float, "prescription_score": float,
                           "other_score": float, "image_name": str, "s3_url": str} }

    Matching is done on filename rather than the full s3_url/reportimgurl string
    because the legacy DB's reportimgurl and the CSV's s3_url may carry different
    bucket/path prefixes for the same physical file.
    """
    if not os.path.exists(csv_path):
        log.warning(
            "Predictions CSV not found at %s -- ALL documents will be skipped "
            "as 'no-prediction-match'. Pass --predictions-csv to point at the right file.",
            csv_path,
        )
        return {}

    lookup: dict[str, dict] = {}
    dupes = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s3_url = (row.get("s3_url") or "").strip()
            if not s3_url:
                continue
            fname = filename_from_key(s3_url)
            if fname in lookup:
                dupes += 1
            lookup[fname] = {
                "image_name":         row.get("image_name"),
                "s3_url":             s3_url,
                "person_id":          row.get("person_id"),
                "healthcase_id":      row.get("healthcase_id"),
                "prediction":         (row.get("prediction") or "").strip(),
                "confidence":         row.get("confidence"),
                "xray_score":         row.get("xray_score"),
                "prescription_score": row.get("prescription_score"),
                "other_score":        row.get("other_score"),
            }

    log.info(
        "Loaded %d prediction rows from %s (%d duplicate filenames overwritten)",
        len(lookup), csv_path, dupes,
    )
    return lookup


def get_prediction_for_key(old_key: str, prediction_lookup: dict[str, dict]) -> dict | None:
    return prediction_lookup.get(filename_from_key(old_key))


# ---------------------------------------------------------------------------
# Validation stats (person_id sanity check) -- purely informational
# ---------------------------------------------------------------------------

_VALIDATION_STATS_SQL = text("""
    WITH joined AS (
        SELECT
            a.id                AS attached_report_id,
            a.healthcase_id,
            b.imagereportid,
            b.person_id         AS image_person_id
        FROM "HealthCase_attachedreporthealthcase" a
        JOIN "BodyVitals_imagereportdetails" b
            ON a.attachedreportid = b.imagereportid
    ),
    validated AS (
        SELECT
            j.*,
            h.person_id                                            AS healthcase_person_id,
            (j.image_person_id = h.person_id)                      AS is_correct
        FROM joined j
        JOIN "HealthCase_healthcase" h
            ON h.id = j.healthcase_id
    )
    SELECT
        COUNT(*)                                                                      AS total_joined_records,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)                                   AS correct_mappings,
        SUM(CASE WHEN NOT is_correct THEN 1 ELSE 0 END)                               AS incorrect_mappings,
        ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pct_correct
    FROM validated;
""")


def log_validation_stats(legacy_engine: Any) -> None:
    """
    Log the person_id validation breakdown (image report's person vs the
    health case's actual person) before migrating. Only 'correct' rows are
    ever fetched/migrated -- this is just visibility into how many rows are
    being excluded and why.
    """
    with legacy_engine.connect() as conn:
        row = conn.execute(_VALIDATION_STATS_SQL).mappings().first()
    if row:
        log.info(
            "== PERSON_ID VALIDATION ==  total_joined=%s | correct=%s | incorrect=%s | "
            "pct_correct=%s%%  (only 'correct' rows are eligible for migration)",
            row["total_joined_records"], row["correct_mappings"],
            row["incorrect_mappings"], row["pct_correct"],
        )


# ---------------------------------------------------------------------------
# Source SQL — HCA joined to BIR (for S3 key) and HealthCase (for validation)
#
# Only rows where BodyVitals_imagereportdetails.person_id matches
# HealthCase_healthcase.person_id are fetched -- i.e. cases where the image
# report actually belongs to the same person as the health case it's
# attached to. Mismatched rows are excluded entirely (never migrated).
# ---------------------------------------------------------------------------

_HCA_FETCH_SQL = text("""
    SELECT
        hca.id,
        hca.attachedreportid,
        hca.attachedreporttitle,
        hca.datetime,
        hca.uploaderrole,
        hca.healthcase_id,
        ir.reportimgurl,
        ir.person_id AS image_person_id,
        h.person_id  AS healthcase_person_id
    FROM "HealthCase_attachedreporthealthcase" hca
    INNER JOIN "BodyVitals_imagereportdetails" ir
        ON ir.imagereportid = hca.attachedreportid
    INNER JOIN "HealthCase_healthcase" h
        ON h.id = hca.healthcase_id
    WHERE ir.reportimgurl IS NOT NULL
      AND ir.reportimgurl <> ''
      AND hca.healthcase_id IS NOT NULL
      AND ir.person_id = h.person_id
    ORDER BY hca.id ASC
""")

_HCA_FETCH_LIMITED_SQL = text("""
    SELECT
        hca.id,
        hca.attachedreportid,
        hca.attachedreporttitle,
        hca.datetime,
        hca.uploaderrole,
        hca.healthcase_id,
        ir.reportimgurl,
        ir.person_id AS image_person_id,
        h.person_id  AS healthcase_person_id
    FROM "HealthCase_attachedreporthealthcase" hca
    INNER JOIN "BodyVitals_imagereportdetails" ir
        ON ir.imagereportid = hca.attachedreportid
    INNER JOIN "HealthCase_healthcase" h
        ON h.id = hca.healthcase_id
    WHERE ir.reportimgurl IS NOT NULL
      AND ir.reportimgurl <> ''
      AND hca.healthcase_id IS NOT NULL
      AND ir.person_id = h.person_id
    ORDER BY hca.id ASC
    LIMIT :lim
""")


# ---------------------------------------------------------------------------
# Reliable lookup: healthcase_id -> (visit_uuid, patient_uuid) via visit table
# ---------------------------------------------------------------------------

def load_healthcase_visit_lookup(new_engine: Any) -> dict[str, tuple[str, str]]:
    """
    healthcase_id -> (visit.id, visit.patient_id), sourced only from rows whose
    legacy_source = 'digiswasthya_database' (i.e. rows migrated by migrate_visit.py
    using HealthCase_healthcase.person_id -- the trustworthy side of the mismatch).
    """
    with new_engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT legacy_id, id::text, patient_id::text
                FROM visit
                WHERE legacy_id IS NOT NULL
                  AND patient_id IS NOT NULL
                  AND legacy_source = :legacy_source
            """),
            {"legacy_source": VISIT_LEGACY_SOURCE},
        ).fetchall()

    mapping = {str(r[0]): (r[1], r[2]) for r in rows}
    log.info(
        "Loaded %d healthcase_id -> (visit_uuid, patient_uuid) mappings from visit table",
        len(mapping),
    )
    return mapping


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_document_record(
    new_doc_id: str,
    new_patient_id: str,
    new_visit_id: str | None,
    storage_key: str,
    storage_url: str,
    content_type: str,
    name: str,
    source: str,
    document_type: str,
    created_at: Any,
    legacy_id: str,
) -> dict:
    return {
        "id":                  new_doc_id,
        "patient_id":          new_patient_id,
        "visit_id":            new_visit_id,
        "name":                name,
        "file_path":           storage_url,
        "storage_key":         storage_key,
        "storage_url":         storage_url,
        "content_type":        content_type,
        "category":            CATEGORY,
        "document_type":       document_type,
        "visibility":          VISIBILITY,
        "is_system_generated": False,
        "is_pinned":           False,
        "file_size_bytes":     None,
        "source":              source,
        "triage_session_id":   None,
        "uploaded_by_user_id": None,
        "deleted_at":          None,
        "legacy_id":           legacy_id,
        "legacy_source":       "Digiswasthya Database",
        "created_at":          created_at,
    }


INSERT_SQL = text("""
    INSERT INTO patientdocument (
        id, patient_id, visit_id, name, file_path,
        storage_key, storage_url, content_type,
        category, document_type, visibility,
        is_system_generated, is_pinned, file_size_bytes,
        source, triage_session_id, uploaded_by_user_id,
        deleted_at, legacy_id, legacy_source, created_at
    ) VALUES (
        :id, :patient_id, :visit_id, :name, :file_path,
        :storage_key, :storage_url, :content_type,
        :category, :document_type, :visibility,
        :is_system_generated, :is_pinned, :file_size_bytes,
        :source, :triage_session_id, :uploaded_by_user_id,
        :deleted_at, :legacy_id, :legacy_source, :created_at
    )
    ON CONFLICT (id) DO NOTHING
""")


def _flush_batch(new_engine: Any, batch: list[dict]) -> None:
    if DRY_RUN:
        log.info("[DRY RUN] Would insert %d patientdocument rows.", len(batch))
        return
    with Session(new_engine) as session:
        session.execute(INSERT_SQL, batch)
        session.commit()


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------

def migrate_healthcase_attached(
    new_engine: Any,
    legacy_engine: Any,
    s3,
    id_gen: Any,
    healthcase_visit_lookup: dict[str, tuple[str, str]],
    prediction_lookup: dict[str, dict],
    already_migrated: set,
    limit: int | None,
) -> dict:
    log_validation_stats(legacy_engine)

    with legacy_engine.connect() as conn:
        if limit is not None:
            rows = conn.execute(_HCA_FETCH_LIMITED_SQL, {"lim": limit}).mappings().all()
        else:
            rows = conn.execute(_HCA_FETCH_SQL).mappings().all()

    with legacy_engine.connect() as conn:
        total_in_db = conn.execute(
            text('SELECT COUNT(*) FROM "HealthCase_attachedreporthealthcase"')
        ).scalar()

    log.info(
        "== SOURCE ==  HealthCase_attachedreporthealthcase total=%d | "
        "fetched (person_id-validated only)=%d (limit=%s)",
        total_in_db, len(rows), limit if limit is not None else "ALL",
    )

    batch: list[dict] = []
    inserted = 0
    skipped_migrated = 0
    skipped_no_visit = 0
    skipped_no_s3_key = 0
    skipped_no_prediction_match = 0
    skipped_prediction_not_allowed = 0
    s3_failures = 0
    errors = 0

    for row in rows:
        old_id        = row.get("id")
        healthcase_id = str(row.get("healthcase_id", ""))
        legacy_id     = f"hca_{old_id}"

        if legacy_id in already_migrated:
            skipped_migrated += 1
            continue

        # Reliable resolution: healthcase_id -> visit -> patient_id.
        # Deliberately NOT using BodyVitals_imagereportdetails.person_id here
        # for patient resolution -- it's only used above (in SQL) to validate
        # that the image report actually belongs to this health case's person.
        visit_info = healthcase_visit_lookup.get(healthcase_id)
        if visit_info is None:
            log.warning(
                "SKIP id=%s -- healthcase_id=%s has no migrated visit "
                "(no reliable patient_id source)", old_id, healthcase_id,
            )
            skipped_no_visit += 1
            continue

        new_visit_id, new_patient_id = visit_info

        img_urls = [k.strip() for k in (row.get("reportimgurl") or "").split(";") if k.strip()]
        if not img_urls:
            log.warning("SKIP id=%s -- no S3 key found in joined BIR row", old_id)
            skipped_no_s3_key += 1
            continue
        old_key = img_urls[0]   # HCA is always single-file per attached report

        # Prediction CSV gate -- more accurate document_type than legacy data.
        pred_row = get_prediction_for_key(old_key, prediction_lookup)
        if pred_row is None:
            log.warning(
                "SKIP id=%s -- filename %s not found in predictions CSV",
                old_id, filename_from_key(old_key),
            )
            skipped_no_prediction_match += 1
            continue

        prediction = pred_row["prediction"]
        if prediction not in ALLOWED_PREDICTIONS:
            log.debug(
                "SKIP id=%s -- prediction=%r not in allowed set, filename=%s",
                old_id, prediction, filename_from_key(old_key),
            )
            skipped_prediction_not_allowed += 1
            continue

        document_type = PREDICTION_TO_DOCUMENT_TYPE[prediction]
        source        = source_from_uploaderrole(row.get("uploaderrole"))
        title         = (row.get("attachedreporttitle") or "").strip()
        fname         = filename_from_key(old_key)
        name          = title if title and title not in (".", "1", "P", "Q", "B", "T", "I") else fname

        try:
            new_s3_key   = build_new_s3_key(new_patient_id, old_key)
            storage_url  = build_storage_url(new_s3_key)
            content_type = content_type_from_key(old_key)

            ok = copy_s3_object(s3, old_key, new_s3_key)
            if not ok:
                s3_failures += 1
                continue

            record = build_document_record(
                new_doc_id=id_gen.next(),
                new_patient_id=new_patient_id,
                new_visit_id=new_visit_id,
                storage_key=new_s3_key,
                storage_url=storage_url,
                content_type=content_type,
                name=name,
                source=source,
                document_type=document_type,
                created_at=row.get("datetime"),
                legacy_id=legacy_id,
            )
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                _flush_batch(new_engine, batch)
                inserted += len(batch)
                log.info("  ... %d rows committed", inserted)
                batch.clear()

        except Exception as e:
            log.error("ERROR id=%s healthcase=%s: %s", old_id, healthcase_id, e)
            errors += 1

    if batch:
        _flush_batch(new_engine, batch)
        inserted += len(batch)

    log.info(
        "=== Migration complete ===  fetched=%d | inserted=%d | "
        "skipped(re-run)=%d | skipped(no-visit)=%d | skipped(no-s3-key)=%d | "
        "skipped(no-prediction-match)=%d | skipped(prediction-not-allowed)=%d | "
        "s3_fail=%d | errors=%d",
        len(rows), inserted, skipped_migrated, skipped_no_visit, skipped_no_s3_key,
        skipped_no_prediction_match, skipped_prediction_not_allowed, s3_failures, errors,
    )
    return {
        "inserted": inserted,
        "skipped_migrated": skipped_migrated,
        "skipped_no_visit": skipped_no_visit,
        "skipped_no_s3_key": skipped_no_s3_key,
        "skipped_no_prediction_match": skipped_no_prediction_match,
        "skipped_prediction_not_allowed": skipped_prediction_not_allowed,
        "s3_failures": s3_failures,
        "errors": errors,
    }


def migrate_documents(limit: int | None = None, predictions_csv: str = DEFAULT_PREDICTIONS_CSV) -> None:
    legacy_engine = get_legacy_engine()
    new_engine    = get_new_engine()

    healthcase_visit_lookup = load_healthcase_visit_lookup(new_engine)
    prediction_lookup       = load_prediction_lookup(predictions_csv)

    already_migrated: set[str] = get_migrated_legacy_ids(new_engine, "patientdocument")
    log.info("Already migrated in new DB: %d patientdocument rows", len(already_migrated))

    s3     = get_s3_client()
    id_gen = SafeIDGenerator(new_engine, table="patientdocument")

    result = migrate_healthcase_attached(
        new_engine, legacy_engine, s3, id_gen,
        healthcase_visit_lookup, prediction_lookup, already_migrated, limit,
    )

    log.info(
        "======== TOTAL ========  inserted=%d | s3_fail=%d | errors=%d",
        result["inserted"], result["s3_failures"], result["errors"],
    )


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_documents(limit: int | None = None, predictions_csv: str = DEFAULT_PREDICTIONS_CSV) -> None:
    if limit is None:
        limit = PREVIEW_SAMPLE_SIZE

    legacy_engine = get_legacy_engine()
    new_engine    = get_new_engine()

    log_validation_stats(legacy_engine)

    healthcase_visit_lookup = load_healthcase_visit_lookup(new_engine)
    prediction_lookup       = load_prediction_lookup(predictions_csv)

    with legacy_engine.connect() as conn:
        rows = conn.execute(_HCA_FETCH_LIMITED_SQL, {"lim": limit}).mappings().all()

    log.info("======== PREVIEW — %d rows (person_id-validated only) ========", len(rows))
    for i, row in enumerate(rows, 1):
        old_id        = row.get("id")
        healthcase_id = str(row.get("healthcase_id", ""))
        visit_info    = healthcase_visit_lookup.get(healthcase_id)

        if visit_info is None:
            log.warning("PREVIEW [%d/%d] id=%s WOULD SKIP: no visit for healthcase_id=%s",
                        i, len(rows), old_id, healthcase_id)
            continue

        new_visit_id, new_patient_id = visit_info
        img_urls = [k.strip() for k in (row.get("reportimgurl") or "").split(";") if k.strip()]
        old_key  = img_urls[0] if img_urls else None

        if not old_key:
            log.warning("PREVIEW [%d/%d] id=%s WOULD SKIP: no S3 key", i, len(rows), old_id)
            continue

        pred_row = get_prediction_for_key(old_key, prediction_lookup)
        if pred_row is None:
            log.warning("PREVIEW [%d/%d] id=%s WOULD SKIP: no prediction match for %s",
                        i, len(rows), old_id, filename_from_key(old_key))
            continue

        prediction = pred_row["prediction"]
        if prediction not in ALLOWED_PREDICTIONS:
            log.info("PREVIEW [%d/%d] id=%s WOULD SKIP: prediction=%r not allowed",
                      i, len(rows), old_id, prediction)
            continue

        log.info(
            "PREVIEW [%d/%d] id=%s  patient=%s  visit=%s  prediction=%s  "
            "(image_person=%s == healthcase_person=%s)\n"
            "  old=%s/%s\n  new=%s/%s",
            i, len(rows), old_id, new_patient_id, new_visit_id, prediction,
            row.get("image_person_id"), row.get("healthcase_person_id"),
            OLD_S3_BUCKET, old_key,
            NEW_S3_BUCKET, build_new_s3_key(new_patient_id, old_key),
        )

    log.info("======== PREVIEW complete -- no rows written ========")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate documents from HealthCase_attachedreporthealthcase (+ BIR for S3 key) -> patientdocument (+ S3), "
                     "restricted to rows where the image report's person matches the health case's person, "
                     "and filtered by prediction CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_documents.py                                   # migrate ALL eligible
  python migrate_documents.py --limit 10                        # first 10 rows
  python migrate_documents.py --preview                         # preview only, no writes
  python migrate_documents.py --predictions-csv ./preds.csv      # custom CSV path
        """,
    )
    parser.add_argument("--limit",   type=int, default=None, metavar="N")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=DEFAULT_PREDICTIONS_CSV,
        metavar="PATH",
        help="Path to prediction CSV (image_name,s3_url,person_id,healthcase_id,prediction,confidence,xray_score,prescription_score,other_score)",
    )
    args = parser.parse_args()

    preview_sample = args.limit if args.limit is not None else PREVIEW_SAMPLE_SIZE
    preview_documents(limit=preview_sample, predictions_csv=args.predictions_csv)

    if args.preview:
        log.info("--preview flag set -- exiting without migrating.")
        sys.exit(0)

    limit_label = str(args.limit) if args.limit is not None else "ALL"
    log.info(
        "Review the PREVIEW logs above. "
        "About to migrate %s document(s). Type 'yes' to proceed.",
        limit_label,
    )
    answer = input("Migrate to new DB? (yes/no): ").strip().lower()
    if answer != "yes":
        log.info("Migration cancelled -- no data written.")
        sys.exit(0)

    migrate_documents(limit=args.limit, predictions_csv=args.predictions_csv)