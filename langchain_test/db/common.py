from datetime import datetime, timezone

from supabase import create_client

from config import SUPABASE_URL, SUPABASE_KEY


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

