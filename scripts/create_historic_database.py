# =========================================================
# File: scripts/create_historic_database.py
# Purpose: Dump the entire speeches table to a local Parquet file.
#          Upload it manually to Google Drive and set PARQUET_URL in .env.
# =========================================================

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from supabase import create_client
from datetime import date
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL, SUPABASE_KEY or SUPABASE_SERVICE_KEY in .env")

supabase_read = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_speeches():
    batch_size = 1000
    offset = 0
    dfs = []

    query = supabase_read.table("speeches").select("*").order("protocol_date", desc=False)

    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if not resp.data:
            break
        dfs.append(pd.DataFrame(resp.data))
        print(f"Fetched {len(resp.data)} rows (total {sum(len(df) for df in dfs)})")
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not dfs:
        raise RuntimeError("No data fetched from Supabase.")

    df = pd.concat(dfs, ignore_index=True)
    df["protocol_date"] = pd.to_datetime(df["protocol_date"], errors="coerce")
    return df


def write_parquet(df: pd.DataFrame, out_dir="data/parquet"):
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    filename = f"speeches_historic_{today}.parquet"
    filepath = os.path.join(out_dir, filename)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression="snappy")

    print(f"Parquet written: {filepath}")
    return filepath, filename


if __name__ == "__main__":
    print("Fetching all historical data (speeches)...")
    df = fetch_all_speeches()
    print(f"Total rows: {len(df)}")

    print("Writing Parquet...")
    local_path, filename = write_parquet(df)

    print("\nDONE!")
    print(f"Upload '{local_path}' to Google Drive, then set in .env:")
    print("PARQUET_URL=<your-public-google-drive-url>")
