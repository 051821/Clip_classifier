"""
db/database.py
--------------
Fetches image metadata from the legacy database.

Queries BOTH source tables that the migration script uses:
  1. BodyVitals_imagereportdetails  (BIR)
  2. HealthCase_attachedreporthealthcase  (HCA)  — includes healthcase_id

Each row returned is a dict with:
  - image_name    : bare filename (e.g. "abc.jpg")
  - s3_key        : the S3 object key  (e.g. "2xsqqgw/image_abc.jpg")
  - s3_url        : full s3:// URL     (e.g. "s3://bucket/2xsqqgw/image_abc.jpg")
  - legacy_id     : "bir_{id}_{idx}"  or  "hca_{id}"
  - legacy_source : "bodyvitals_imagereport"  or  "healthcase_attached"
  - person_id     : BIR -> ir.person_id; HCA -> h.person_id (validated, see below)
  - healthcase_id : populated for HCA rows; None for BIR rows

HCA rows are only fetched when ir.person_id = h.person_id, i.e. the image
report's person actually matches the health case's own person. This is the
same check used in the validation query / migrate_documents.py's SQL filter,
so a row that appears in this CSV is guaranteed to already be
person-ID-correct -- no separately mismatched rows get written out.
"""

from typing import Generator
from sqlalchemy import create_engine, text
from config.settings import DB_URL, S3_BUCKET_NAME
from utils.logger import get_logger

log = get_logger("db")

# ── SQL ───────────────────────────────────────────────────────────────────────

_BIR_SQL = """
    SELECT
        ir.id,
        ir.reportimgurl,
        ir.reporttitle,
        ir.person_id
    FROM "BodyVitals_imagereportdetails" ir
    WHERE ir.reportimgurl IS NOT NULL
      AND ir.reportimgurl <> ''
      AND ir.person_id    IS NOT NULL
    ORDER BY ir.id ASC
"""

# HCA rows are additionally validated against HealthCase_healthcase:
# only rows where the image report's person_id actually matches the
# health case's own person_id are fetched. This mirrors the check from
# the validation query (ir.person_id = h.person_id) and keeps
# migrate_documents.py's SQL-level filter and this CSV in agreement.
_HCA_SQL = """
    SELECT
        hca.id,
        hca.healthcase_id,
        hca.attachedreporttitle,
        hca.uploaderrole,
        ir.reportimgurl,
        h.person_id
    FROM "HealthCase_attachedreporthealthcase" hca
    INNER JOIN "BodyVitals_imagereportdetails" ir
        ON ir.imagereportid = hca.attachedreportid
    INNER JOIN "HealthCase_healthcase" h
        ON h.id = hca.healthcase_id
    WHERE ir.reportimgurl IS NOT NULL
      AND ir.reportimgurl <> ''
      AND ir.person_id    IS NOT NULL
      AND hca.healthcase_id IS NOT NULL
      AND ir.person_id = h.person_id
    ORDER BY hca.id ASC
"""


def _filename(key: str) -> str:
    """Return bare filename from an S3 key."""
    return key.strip().split("/")[-1]


def _build_s3_url(key: str) -> str:
    return f"s3://{S3_BUCKET_NAME}/{key.strip()}"


def fetch_images(engine) -> Generator[dict, None, None]:
    """
    Yields one dict per image to classify.
    Handles both BIR (semicolon-separated keys) and HCA (single key) sources.
    """
    with engine.connect() as conn:

        # ── Source 1: BIR ─────────────────────────────────────────────────
        log.info("Fetching from BodyVitals_imagereportdetails ...")
        bir_rows = conn.execute(
            text(_BIR_SQL)
        ).mappings().all()
        log.info("  BIR rows fetched: %d", len(bir_rows))

        for row in bir_rows:
            keys = [k.strip() for k in (row["reportimgurl"] or "").split(";") if k.strip()]
            for idx, key in enumerate(keys):
                yield {
                    "image_name":    _filename(key),
                    "s3_key":        key,
                    "s3_url":        _build_s3_url(key),
                    "legacy_id":     f"bir_{row['id']}_{idx}",
                    "legacy_source": "bodyvitals_imagereport",
                    "person_id":     str(row["person_id"]),
                    "healthcase_id": None,
                }

        # ── Source 2: HCA ─────────────────────────────────────────────────
        log.info("Fetching from HealthCase_attachedreporthealthcase ...")
        hca_rows = conn.execute(
            text(_HCA_SQL)
        ).mappings().all()
        log.info("  HCA rows fetched: %d", len(hca_rows))

        for row in hca_rows:
            keys = [k.strip() for k in (row["reportimgurl"] or "").split(";") if k.strip()]
            if not keys:
                continue
            key = keys[0]  # HCA is always single-file
            yield {
                "image_name":    _filename(key),
                "s3_key":        key,
                "s3_url":        _build_s3_url(key),
                "legacy_id":     f"hca_{row['id']}",
                "legacy_source": "healthcase_attached",
                "person_id":     str(row["person_id"]),
                "healthcase_id": str(row["healthcase_id"]),
            }


def get_engine():
    """Create and return a SQLAlchemy engine."""
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        log.info("Database engine created: %s:%s/%s", 
                 DB_URL.split("@")[-1].split("/")[0],  # host only, no creds
                 "", "")
        return engine
    except Exception as e:
        log.error("Failed to create database engine: %s", e)
        raise