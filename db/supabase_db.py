import urllib.request
import json
from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY
from utils.logger import get_logger

log = get_logger("supabase_db")

try:
    from supabase import create_client, Client
    log.info("Using official Supabase Python SDK.")
except ModuleNotFoundError:
    log.warning("Supabase SDK not installed. Falling back to native urllib HTTP client.")

    class SupabaseResponse:
        def __init__(self, data):
            self.data = data

    class SupabaseQueryBuilder:
        def __init__(self, url, headers, table_name):
            self.url = url
            self.headers = headers
            self.table_name = table_name
            self.select_str = "*"

        def select(self, select_str):
            # Clean up whitespace/newlines
            self.select_str = "".join(select_str.split())
            return self

        @property
        def not_(self):
            # Chains to is_
            return self

        def is_(self, column, val):
            # Chains .is_("storage_url", "null") -> storage_url=not.is.null
            return self

        def execute(self):
            query_url = f"{self.url}/rest/v1/{self.table_name}?select={self.select_str}&storage_url=not.is.null"
            req = urllib.request.Request(query_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return SupabaseResponse(data)

    class Client:
        def __init__(self, url, key):
            self.url = url
            self.headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }

        def table(self, table_name):
            return SupabaseQueryBuilder(self.url, self.headers, table_name)

    def create_client(url, key):
        return Client(url, key)


def get_supabase_client() -> Client:
    """Create and return a Supabase Client."""
    url = SUPABASE_URL.strip()
    key = SUPABASE_SERVICE_KEY.strip()
    log.info("Initializing Supabase client: %s", url)
    return create_client(url, key)


def fetch_supabase_documents(client: Client) -> list[dict]:
    """
    Fetch all rows from 'patientdocument' table where storage_url is not null.
    """
    log.info("Fetching document metadata from patientdocument table...")
    try:
        response = (
            client.table("patientdocument")
            .select("""
                id,
                patient_id,
                name,
                document_type,
                storage_key,
                storage_url,
                file_path,
                content_type,
                created_at,
                visit_id
            """)
            .not_.is_("storage_url", "null")
            .execute()
        )
        records = response.data if hasattr(response, "data") else []
        log.info("Successfully fetched %d records from patientdocument", len(records))
        return records
    except Exception as e:
        log.error("Failed to fetch documents from Supabase: %s", e)
        raise
