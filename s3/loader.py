"""
s3/loader.py
------------
Downloads images from S3 directly into RAM.
Images are NEVER written to disk.

Flow:
  S3 → bytes → BytesIO → PIL Image → CLIP
"""

from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import ClientError

from config.settings import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME,
)
from utils.logger import get_logger

log = get_logger("s3")


import re
import urllib.request
from urllib.parse import urlparse

def get_s3_client():
    """Create and return a boto3 S3 client."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def parse_s3_url(url: str) -> tuple[str, str] | None:
    """
    Parses an S3 URL/URI and returns (bucket, key).
    Supports:
      - s3://bucket/key
      - https://bucket.s3.amazonaws.com/key
      - https://bucket.s3-region.amazonaws.com/key
      - https://bucket.s3.region.amazonaws.com/key
      - https://s3.amazonaws.com/bucket/key
      - https://s3-region.amazonaws.com/bucket/key
      - https://s3.region.amazonaws.com/bucket/key
    """
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    if parsed.scheme in ("http", "https"):
        host = parsed.netloc.lower()
        path = parsed.path.lstrip("/")

        # Case 1: bucket.s3.amazonaws.com or bucket.s3-region.amazonaws.com
        # or bucket.s3.region.amazonaws.com
        m = re.match(r"^([^.]+)\.s3(?:[-.][a-z0-9-]+)?\.amazonaws\.com$", host)
        if m:
            bucket = m.group(1)
            return bucket, path

        # Case 2: s3.amazonaws.com/bucket/key or s3-region.amazonaws.com/bucket/key
        # or s3.region.amazonaws.com/bucket/key
        if re.match(r"^s3(?:[-.][a-z0-9-]+)?\.amazonaws\.com$", host):
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

    return None


def load_image_from_url(url: str, headers: dict | None = None) -> Image.Image | None:
    """Download an image from a URL and return a PIL Image."""
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            image_bytes = response.read()
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        log.error("Failed to load image from URL=%s: %s", url, e)
        return None


def _fetch_from_s3(s3_client, bucket: str, key: str) -> Image.Image | None:
    """Download an image from S3 and return a PIL Image."""
    try:
        if s3_client is None:
            s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            log.warning("S3 object not found: s3://%s/%s", bucket, key)
        else:
            log.error("S3 ClientError for bucket=%s key=%s: %s", bucket, key, e)
        return None
    except Exception as e:
        log.error("Failed to load image from S3 bucket=%s key=%s: %s", bucket, key, e)
        return None


def load_image(uri_or_key: str, s3_client=None) -> Image.Image | None:
    """
    Unified function to load an image dynamically based on format:
    - S3 URI (s3://...)
    - S3 HTTPS URL
    - Supabase storage URL (with automatic headers if configured)
    - Standard HTTP/HTTPS URL
    - Fallback to S3 key (using configured S3_BUCKET_NAME)
    """
    if not uri_or_key:
        log.warning("Empty image URL or key provided.")
        return None

    uri_or_key_str = str(uri_or_key).strip()

    # 1. Check if S3 URI or S3 HTTPS URL
    s3_info = parse_s3_url(uri_or_key_str)
    if s3_info:
        bucket, key = s3_info
        log.debug("Detected S3 location: bucket=%s, key=%s", bucket, key)
        return _fetch_from_s3(s3_client, bucket, key)

    # 2. Check if general HTTP/HTTPS URL
    if uri_or_key_str.startswith(("http://", "https://")):
        headers = {"User-Agent": "Mozilla/5.0"}
        # Inject Supabase service key if URL points to supabase.co and key is defined
        from config.settings import SUPABASE_SERVICE_KEY
        if "supabase.co" in uri_or_key_str.lower() and SUPABASE_SERVICE_KEY:
            clean_key = SUPABASE_SERVICE_KEY.strip()
            if clean_key:
                headers["Authorization"] = f"Bearer {clean_key}"
                headers["apikey"] = clean_key
                log.debug("Injecting Supabase authentication headers for URL: %s", uri_or_key_str)

        return load_image_from_url(uri_or_key_str, headers=headers)

    # 3. Fallback: treat as raw S3 key in the default bucket
    from config.settings import S3_BUCKET_NAME
    if S3_BUCKET_NAME:
        log.debug("Treating as raw S3 key in bucket %s: %s", S3_BUCKET_NAME, uri_or_key_str)
        return _fetch_from_s3(s3_client, S3_BUCKET_NAME, uri_or_key_str)

    log.error("Could not load image: %s (no S3 bucket or invalid URL)", uri_or_key_str)
    return None


def load_image_from_s3(s3_client, s3_key: str) -> Image.Image | None:
    """
    Download an image from default S3 bucket. Kept for backward compatibility.
    """
    from config.settings import S3_BUCKET_NAME
    return _fetch_from_s3(s3_client, S3_BUCKET_NAME, s3_key)
