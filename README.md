# CLIP Medical Image Classification Pipeline

Classifies ~70,000 medical images stored in S3 using OpenAI CLIP.
Outputs a `prediction.csv` that the migration script reads as a lookup table.

Images are **never saved to disk** — downloaded into RAM, classified, discarded.

---

## Project Structure

```
clip_classifier/
├── main.py                  ← Entry point. Run this.
├── config/
│   └── settings.py          ← Reads all config from environment variables
├── db/
│   └── database.py          ← Fetches image metadata from legacy DB
├── s3/
│   └── loader.py            ← Downloads images from S3 into memory
├── classifier/
│   └── clip_classifier.py   ← CLIP model + inference logic
├── writer/
│   └── csv_writer.py        ← Streams results to CSV
├── utils/
│   └── logger.py            ← Logging setup
├── requirements.txt
├── Dockerfile
├── .env.example             ← Copy to .env and fill in your credentials
└── README.md
```

---

## Output CSV

`prediction.csv` contains one row per image:

| Column               | Example                          | Notes                              |
|----------------------|----------------------------------|------------------------------------|
| `image_name`         | `abc.jpg`                        | Bare filename                      |
| `s3_url`             | `s3://bucket/path/abc.jpg`       | Full S3 URL                        |
| `legacy_id`          | `bir_42_0` or `hca_17`          | Links back to source DB row        |
| `legacy_source`      | `bodyvitals_imagereport`         |                                    |
| `healthcase_id`      | `hca_123` or *(empty)*          | Populated for HCA rows only        |
| `prediction`         | `Prescription / Document`        | Final label                        |
| `confidence`         | `0.9821`                         | Softmax probability of top class   |
| `xray_score`         | `0.0041`                         | Raw score for X-ray class          |
| `prescription_score` | `0.9821`                         | Raw score for Prescription class   |
| `other_score`        | `0.0138`                         | Raw score for Other class          |

**Prediction values:**
- `X-ray / Sonography`
- `Prescription / Document`
- `Other`
- `Review Needed` — confidence below threshold (default 0.65)
- `S3_ERROR` — image could not be downloaded
- `ERROR` — CLIP inference failed

---

## Setup

### Option A — Run locally

```bash
# 1. Clone / copy the project
cd clip_classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Fill in .env with values from your manager

# 5. (Optional) Test with 10 images first
echo "LIMIT=10" >> .env

# 6. Run
python main.py
```

### Option B — Run with Docker

```bash
# Build image (downloads CLIP weights into the image — ~1.7 GB)
docker build -t clip-classifier .

# Run — pass all credentials at runtime via --env-file
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/output \
  clip-classifier

# prediction.csv will be in ./output/prediction.csv
```

---

## Testing before full run

Set `LIMIT=10` in your `.env` to process only 10 images and verify the output CSV looks correct before running all 70k images.

---

## Tuning

| Setting                | Default | What it does                                        |
|------------------------|---------|-----------------------------------------------------|
| `CONFIDENCE_THRESHOLD` | `0.65`  | Below this → "Review Needed" instead of a class    |
| `BATCH_SIZE`           | `16`    | Images per CLIP batch (increase for GPU)            |
| `LIMIT`                | *(all)* | Process only N images (useful for testing)          |

---

## How the migration script uses this CSV

The migration script does **not** run the ML model.
It simply loads `prediction.csv` as a lookup:

```python
import csv

predictions = {}
with open("prediction.csv") as f:
    for row in csv.DictReader(f):
        predictions[row["image_name"]] = row

# During migration:
pred = predictions.get(filename)
if pred and pred["prediction"] == "Prescription / Document":
    # migrate this image
```

---

## What gets logged

- `classifier.log` — full DEBUG log (every image)
- Console (stdout) — INFO level (progress, summary, errors)

---

## Credentials required from manager

| Variable               | Description                          |
|------------------------|--------------------------------------|
| `DB_HOST`              | Legacy database host                 |
| `DB_PORT`              | Database port (usually 5432)         |
| `DB_NAME`              | Database name                        |
| `DB_USER`              | Database username                    |
| `DB_PASSWORD`          | Database password                    |
| `AWS_ACCESS_KEY_ID`    | AWS access key                       |
| `AWS_SECRET_ACCESS_KEY`| AWS secret key                       |
| `AWS_REGION`           | e.g. `ap-south-1`                   |
| `S3_BUCKET_NAME`       | Source S3 bucket name                |

## change pipeline mode from testing to production for full run
## can also be done without docker