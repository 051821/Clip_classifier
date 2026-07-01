"""
config/settings.py
------------------
All configuration is read from environment variables.
No credentials are hardcoded here.

Your manager will provide:
  - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
  - S3_BUCKET_NAME
  - OUTPUT_CSV_PATH  (where to write prediction.csv)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = os.environ.get("DB_PORT",     "5432")
DB_NAME     = os.environ.get("DB_NAME",     "legacy_db")
DB_USER     = os.environ.get("DB_USER",     "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    DB_URL = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME        = os.environ.get("S3_BUCKET_NAME", "")

OUTPUT_CSV_PATH = os.environ.get("OUTPUT_CSV_PATH", "prediction.csv")


CLIP_MODEL_NAME     = "openai/clip-vit-base-patch32"
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))
BATCH_SIZE           = int(os.environ.get("BATCH_SIZE", "16"))

# Set to a number (e.g. 100) to process only that many images — useful for testing.
# Leave empty or unset to process ALL images.
LIMIT = os.environ.get("LIMIT")
LIMIT = int(LIMIT) if LIMIT else None

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "production").lower()


SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://okqyofwzjohjdgczbnro.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get(
    "SUPABASE_SERVICE_KEY",
)
