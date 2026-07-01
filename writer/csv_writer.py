"""
writer/csv_writer.py
--------------------
Writes classification results to a CSV file.

Output columns:
  image_name, s3_url, legacy_id, legacy_source, healthcase_id,
  prediction, confidence, xray_score, prescription_score, other_score

Re-runs APPEND to an existing output file instead of overwriting it.
Use get_existing_legacy_ids() before the run to find out which rows
were already written, so the caller can skip them and avoid duplicates.
"""

import csv
import os
from utils.logger import get_logger

log = get_logger("csv_writer")

FIELDNAMES = [
    "image_name",
    "s3_url",
    "legacy_id",
    "legacy_source",
    "healthcase_id",
    "prediction",
    "confidence",
    "xray_score",
    "prescription_score",
    "other_score",
]


def get_existing_legacy_ids(output_path: str, key_field: str = "legacy_id") -> set:
    """
    Reads legacy_ids already present in an existing output CSV,
    so a re-run can skip rows that were already written.
    Returns an empty set if the file doesn't exist or is empty.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row[key_field] for row in reader if row.get(key_field)}


class CSVWriter:
    """
    Opens the output CSV and streams rows as they are produced.
    Appends to an existing file (without rewriting the header) if one
    is already present with content; otherwise creates a fresh file
    with a header row.

    Call .write_row() for each result, then .close() when done.
    Use as a context manager for automatic cleanup.

    Example:
        with CSVWriter("prediction.csv") as writer:
            writer.write_row({...})
    """

    def __init__(self, output_path: str, fieldnames: list[str] = None):
        self.output_path = output_path
        self.fieldnames = fieldnames if fieldnames else FIELDNAMES
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        file_exists_with_content = (
            os.path.exists(output_path) and os.path.getsize(output_path) > 0
        )
        mode = "a" if file_exists_with_content else "w"

        self._file   = open(output_path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")

        if not file_exists_with_content:
            self._writer.writeheader()

        self._count  = 0
        log.info("CSV writer opened in %r mode: %s", mode, output_path)

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._count += 1
        # Flush every 100 rows so partial results are saved on crash
        if self._count % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        log.info("CSV writer closed. Total rows written this run: %d -> %s", self._count, self.output_path)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()