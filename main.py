"""
main.py
-------
Entry point for the CLIP classification pipeline.

What this does:
  1. Connects to the legacy database.
  2. Fetches image metadata from both BIR and HCA tables.
  3. Deduplicates by s3_key — BIR and HCA rows often point to the same image.
     CLIP inference runs ONCE per unique s3_key. Duplicate rows reuse the
     cached prediction and get their own CSV row with their own legacy_id /
     healthcase_id — so the migration script still has a row for every record.
  4. Loads each unique image from S3 directly into memory (never saves to disk).
  5. Runs CLIP inference on each unique image.
  6. Writes one CSV row per DB record (including duplicates with cached results).

Run:
  python main.py

Environment variables required (see config/settings.py):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
  S3_BUCKET_NAME
  OUTPUT_CSV_PATH   (default: prediction.csv)
  LIMIT             (optional: process only N images for testing)
"""

import time
from collections import defaultdict

from config.settings import OUTPUT_CSV_PATH, LIMIT, PIPELINE_MODE
from db.database import get_engine, fetch_images
from s3.loader import get_s3_client, load_image
from classifier.clip_classifier import CLIPClassifier
from writer.csv_writer import CSVWriter, get_existing_row_keys
from utils.logger import get_logger

log = get_logger("main")

# Sentinel used to cache s3_keys that failed to load,
# so we don't retry them for every duplicate row.
_S3_ERROR_RESULT = {
    "prediction":         "S3_ERROR",
    "confidence":         0.0,
    "xray_score":         0.0,
    "prescription_score": 0.0,
    "other_score":        0.0,
}


TESTING_FIELDNAMES = [
    "id",
    "patient_id",
    "name",
    "document_type",
    "storage_key",
    "storage_url",
    "file_path",
    "content_type",
    "created_at",
    "visit_id",
    "prediction",
    "confidence",
    "xray_score",
    "prescription_score",
    "other_score",
]


def run_pipeline() -> None:
    start_time = time.time()

    log.info("=" * 60)
    log.info("CLIP Classification Pipeline — Starting")
    log.info("Mode         : %s", PIPELINE_MODE.upper())
    log.info("Output CSV   : %s", OUTPUT_CSV_PATH)
    log.info("Limit        : %s", LIMIT if LIMIT else "ALL")
    log.info("=" * 60)

    # ── Step 1: Load CLIP model (done once) ──────────────────────────────────
    classifier = CLIPClassifier()

    stats = defaultdict(int)

    # Cache: key → result dict.
    prediction_cache: dict[str, dict] = {}

    if PIPELINE_MODE == "testing":
        # ── TESTING MODE (Supabase) ──────────────────────────────────────────
        from db.supabase_db import get_supabase_client, fetch_supabase_documents

        supabase_client = get_supabase_client()
        documents = fetch_supabase_documents(supabase_client)

        with CSVWriter(OUTPUT_CSV_PATH, fieldnames=TESTING_FIELDNAMES) as csv_writer:
            for doc in documents:
                if LIMIT and stats["total"] >= LIMIT:
                    log.info("Reached testing limit of %d records. Stopping.", LIMIT)
                    break

                stats["total"] += 1
                storage_url = doc.get("storage_url")

                if not storage_url:
                    log.debug("[%d] Skip: Empty storage_url for doc id %s", stats["total"], doc.get("id"))
                    stats["skipped_empty_url"] += 1
                    continue

                # Skip any records whose storage_url points to Amazon S3
                if "amazonaws.com" in storage_url.lower() or "s3://" in storage_url.lower():
                    log.debug("[%d] SKIP (Amazon S3): %s", stats["total"], storage_url)
                    stats["skipped_s3"] += 1
                    continue

                # Download only the images whose storage_url contains supabase.co
                if "supabase.co" not in storage_url.lower():
                    log.debug("[%d] Skip: Non-Supabase URL in testing mode: %s", stats["total"], storage_url)
                    stats["skipped_non_supabase"] += 1
                    continue

                # Cache check
                if storage_url in prediction_cache:
                    stats["cache_hits"] += 1
                    log.debug(
                        "[%d] CACHE HIT  id=%s  url=%s",
                        stats["total"], doc.get("id"), storage_url,
                    )
                    csv_writer.write_row({**doc, **prediction_cache[storage_url]})
                    continue

                stats["unique_images"] += 1

                # Download image using the loader (which uses urllib.request for URLs)
                pil_image = load_image(storage_url)

                if pil_image is None:
                    stats["failed_loads"] += 1
                    log.warning(
                        "[%d] LOAD FAIL  id=%s  url=%s",
                        stats["total"], doc.get("id"), storage_url,
                    )
                    prediction_cache[storage_url] = _S3_ERROR_RESULT
                    csv_writer.write_row({**doc, **_S3_ERROR_RESULT})
                    continue

                # Run CLIP inference
                result = classifier.classify(pil_image)

                # Cache prediction
                prediction_cache[storage_url] = result

                # Update stats
                if result["prediction"] == "ERROR":
                    stats["inference_failed"] += 1
                elif result["prediction"] == "Review Needed":
                    stats["review_needed"] += 1
                else:
                    stats[f"class_{result['prediction']}"] += 1

                csv_writer.write_row({**doc, **result})

                # Progress log every 100 images
                if stats["unique_images"] % 100 == 0:
                    elapsed = time.time() - start_time
                    rate    = stats["unique_images"] / elapsed
                    log.info(
                        "Progress: %d unique | %d total rows | %.1f img/s | elapsed: %.0fs",
                        stats["unique_images"], stats["total"], rate, elapsed,
                    )

        # Print Testing Summary
        elapsed = time.time() - start_time
        log.info("")
        log.info("=" * 60)
        log.info("Testing Pipeline Complete")
        log.info("=" * 60)
        log.info("Total time              : %.1f seconds", elapsed)
        log.info("Output CSV              : %s", OUTPUT_CSV_PATH)
        log.info("=" * 60)

    else:
        # ── PRODUCTION MODE (Postgres / S3 / Legacy) ──────────────────────────
        db_engine = get_engine()
        s3_client = get_s3_client()

        already_written = get_existing_row_keys(OUTPUT_CSV_PATH)
        if already_written:
            log.info("Resuming: %d rows already in %s — these will be skipped.",
                      len(already_written), OUTPUT_CSV_PATH)

        with CSVWriter(OUTPUT_CSV_PATH) as csv_writer:
            for meta in fetch_images(db_engine):
                if LIMIT and stats["total"] >= LIMIT:
                    log.info("Reached testing limit of %d records. Stopping.", LIMIT)
                    break

                row_key = f"{meta['s3_url']}|{meta['person_id']}|{meta.get('healthcase_id', '')}"
                if row_key in already_written:
                    stats["already_done"] += 1
                    continue

                stats["total"] += 1
                s3_key = meta["s3_key"]

                # ── Cache hit: same image seen before ─────────────────────────
                if s3_key in prediction_cache:
                    stats["cache_hits"] += 1
                    log.debug(
                        "[%d] CACHE HIT  legacy_id=%s  key=%s",
                        stats["total"], meta["legacy_id"], s3_key,
                    )
                    csv_writer.write_row({**meta, **prediction_cache[s3_key]})
                    continue

                # ── Cache miss: first time seeing this s3_key ─────────────────
                stats["unique_images"] += 1

                # Download image into RAM
                pil_image = load_image(s3_key, s3_client)

                if pil_image is None:
                    stats["s3_failed"] += 1
                    log.warning(
                        "[%d] LOAD FAIL  legacy_id=%s  key=%s",
                        stats["total"], meta["legacy_id"], s3_key,
                    )
                    prediction_cache[s3_key] = _S3_ERROR_RESULT
                    csv_writer.write_row({**meta, **_S3_ERROR_RESULT})
                    continue

                # Run CLIP inference
                result = classifier.classify(pil_image)

                # Cache the prediction keyed by s3_key
                prediction_cache[s3_key] = result

                # Update stats
                if result["prediction"] == "ERROR":
                    stats["inference_failed"] += 1
                elif result["prediction"] == "Review Needed":
                    stats["review_needed"] += 1
                else:
                    stats[f"class_{result['prediction']}"] += 1

                csv_writer.write_row({**meta, **result})

                # Progress log every 500 unique images processed
                if stats["unique_images"] % 500 == 0:
                    elapsed = time.time() - start_time
                    rate    = stats["unique_images"] / elapsed
                    log.info(
                        "Progress: %d unique | %d total rows | %.1f img/s | elapsed: %.0fs",
                        stats["unique_images"], stats["total"], rate, elapsed,
                    )

        # Print Production Summary
        elapsed = time.time() - start_time
        log.info("")
        log.info("=" * 60)
        log.info("Pipeline Complete")
        log.info("=" * 60)
        log.info("Total CSV rows written  : %d", stats["total"])
        log.info("  Skipped (already done)   : %d", stats["already_done"])
        log.info("  Unique images (CLIP ran) : %d", stats["unique_images"])
        log.info("  Duplicate rows (cached)  : %d", stats["cache_hits"])
        log.info("S3 failures             : %d", stats["s3_failed"])
        log.info("Inference failures      : %d", stats["inference_failed"])
        log.info("Review needed           : %d", stats["review_needed"])
        log.info("X-ray / Sonography      : %d", stats["class_X-ray / Sonography"])
        log.info("Prescription / Document : %d", stats["class_Prescription / Document"])
        log.info("Other                   : %d", stats["class_Other"])
        log.info("Total time              : %.1f seconds (%.2f min)", elapsed, elapsed / 60)
        log.info("Output CSV              : %s", OUTPUT_CSV_PATH)
        log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()