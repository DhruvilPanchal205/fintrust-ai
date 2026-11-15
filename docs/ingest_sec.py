"""FinTrust AI - Phase 1 ingestion from SEC EDGAR.

What it does, for every company in companies.csv:
  1. Looks up the company's CIK number from the SEC ticker file.
  2. Downloads the company facts JSON (all reported XBRL numbers).
  3. Downloads the submissions JSON (list of filings).
  4. Downloads the last N 10-K documents and extracts Item 7 (MD&A) text.
  5. Saves every raw file in data/bronze/<source>/run_date=YYYY-MM-DD/.
  6. Loads raw tables into DuckDB and logs every step in raw.pipeline_runs.

Setup:
  pip install requests pandas duckdb beautifulsoup4 lxml
  export SEC_USER_AGENT="Your Name your.email@example.com"   # required by the SEC

Run:
  python ingest_sec.py --starter-only          # 10 companies, for development
  python ingest_sec.py                         # all companies
  python ingest_sec.py --skip-filings          # numbers only, no 10-K text

Re-running on the same day skips files already downloaded, so a failed run
can simply be started again.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"


# --------------------------------------------------------------------------
# Polite HTTP client: User-Agent, rate limit, retries
# --------------------------------------------------------------------------
class SecClient:
    def __init__(self, user_agent, max_per_second=8):
        # SEC's stated limit is 10 requests/second; staying under it is polite.
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / max_per_second
        self._last = 0.0

    def get(self, url, retries=4):
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                error = str(exc)
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                error = f"HTTP {resp.status_code}"
            time.sleep(2 ** attempt)  # back off: 1s, 2s, 4s, 8s
        raise RuntimeError(f"Failed after {retries} attempts: {url} ({error})")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def pad_cik(cik):
    return str(int(cik)).zfill(10)


def read_companies(path, starter_only):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if starter_only:
        rows = [r for r in rows if r.get("starter", "").strip().lower() == "yes"]
    return rows


def resolve_ciks(client, companies, bronze_dir):
    """Look up CIKs from the SEC's official ticker file instead of typing them by hand."""
    path = bronze_dir / "sec_tickers" / "company_tickers.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.get(TICKERS_URL).content)
    lookup = {v["ticker"].upper(): v for v in json.loads(path.read_text()).values()}
    missing = []
    for c in companies:
        hit = lookup.get(c["ticker"].upper().replace(".", "-"))
        if hit:
            c["cik"] = pad_cik(hit["cik_str"])
            c["sec_title"] = hit["title"]
        else:
            missing.append(c["ticker"])
    return missing


def download_json(client, url, path):
    """Save raw JSON to bronze; skip if already downloaded today."""
    if path.exists():
        return json.loads(path.read_text()), "skipped_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    path.write_bytes(resp.content)
    return resp.json(), "downloaded"


def flatten_facts(facts_json, ticker, run_date):
    """Turn the nested company facts JSON into one row per reported value (lossless)."""
    cik = pad_cik(facts_json["cik"])
    rows = []
    for taxonomy, concepts in facts_json.get("facts", {}).items():
        for concept, body in concepts.items():
            for unit, observations in body.get("units", {}).items():
                for o in observations:
                    rows.append({
                        "cik": cik, "ticker": ticker, "taxonomy": taxonomy,
                        "concept": concept, "label": body.get("label"), "unit": unit,
                        "value": o.get("val"), "period_start": o.get("start"),
                        "period_end": o.get("end"), "fiscal_year": o.get("fy"),
                        "fiscal_period": o.get("fp"), "form": o.get("form"),
                        "filed": o.get("filed"), "accession": o.get("accn"),
                        "frame": o.get("frame"), "run_date": run_date,
                    })
    return rows


def recent_10k_filings(submissions_json, n_years):
    """Pick the most recent original 10-K filings (not amendments)."""
    recent = submissions_json["filings"]["recent"]
    picked = []
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            picked.append({
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            })
    picked.sort(key=lambda r: r["filing_date"], reverse=True)
    return picked[:n_years]


def extract_item7(html):
    """Extract Item 7 (MD&A) text from a 10-K HTML document.

    The table of contents also mentions 'Item 7', so we try every start match
    and keep the longest section that ends at Item 7A or Item 8.
    Returns None when no section is found, so it can be logged as a quality issue.
    """
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "lxml").get_text("\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    starts = [m.start() for m in re.finditer(r"item\s*7\s*[.:\-—]?\s*management", text, re.I)]
    best = ""
    for s in starts:
        end = re.search(r"item\s*7a\s*[.:\-—]|item\s*8\s*[.:\-—]?\s*financial\s+statements",
                        text[s + 20:], re.I)
        section = text[s: s + 20 + end.start()] if end else ""
        if len(section) > len(best):
            best = section
    best = re.sub(r"\n\s*\n+", "\n\n", best).strip()
    return best if len(best) > 2000 else None  # shorter than this is almost always a TOC line


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run(args):
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent or "@" not in user_agent:
        sys.exit('Set SEC_USER_AGENT first, e.g. export SEC_USER_AGENT="Your Name you@example.com"')

    run_id = str(uuid.uuid4())
    run_date = dt.date.today().isoformat()
    bronze = Path(args.data_dir) / "bronze"
    client = SecClient(user_agent)
    log = []

    def record(source, ticker, status, rows=None, error=None, started=None):
        log.append({"run_id": run_id, "run_date": run_date, "source": source, "ticker": ticker,
                    "status": status, "rows": rows, "error": error,
                    "started_at": started, "finished_at": dt.datetime.now().isoformat()})
        print(f"[{status:>16}] {source:<16} {ticker or ''} {error or ''}")

    companies = read_companies(args.companies, args.starter_only)
    missing = resolve_ciks(client, companies, bronze)
    for t in missing:
        record("cik_lookup", t, "failed", error="ticker not found in SEC ticker file")
    companies = [c for c in companies if "cik" in c]

    fact_rows, filing_rows = [], []
    for c in companies:
        t, cik = c["ticker"], c["cik"]

        started = dt.datetime.now().isoformat()
        try:
            facts, status = download_json(client, FACTS_URL.format(cik=cik),
                                          bronze / "sec_companyfacts" / f"run_date={run_date}" / f"CIK{cik}.json")
            rows = flatten_facts(facts, t, run_date)
            fact_rows.extend(rows)
            record("companyfacts", t, status, rows=len(rows), started=started)
        except Exception as exc:  # keep going; one bad company should not stop the run
            record("companyfacts", t, "failed", error=str(exc), started=started)

        if args.skip_filings:
            continue

        started = dt.datetime.now().isoformat()
        try:
            subs, status = download_json(client, SUBMISSIONS_URL.format(cik=cik),
                                         bronze / "sec_submissions" / f"run_date={run_date}" / f"CIK{cik}.json")
            c["sic"], c["sic_description"] = subs.get("sic"), subs.get("sicDescription")
            c["fiscal_year_end"] = subs.get("fiscalYearEnd")
            filings = recent_10k_filings(subs, args.years)
            record("submissions", t, status, rows=len(filings), started=started)
        except Exception as exc:
            record("submissions", t, "failed", error=str(exc), started=started)
            continue

        for f in filings:
            started = dt.datetime.now().isoformat()
            acc_nodash = f["accession"].replace("-", "")
            html_path = bronze / "sec_10k_html" / cik / f"{acc_nodash}_{f['primary_doc']}"
            text_path = bronze / "sec_10k_item7" / cik / f"{acc_nodash}.txt"
            try:
                if not html_path.exists():
                    html_path.parent.mkdir(parents=True, exist_ok=True)
                    url = DOC_URL.format(cik_int=int(cik), acc_nodash=acc_nodash, doc=f["primary_doc"])
                    html_path.write_bytes(client.get(url).content)
                item7 = extract_item7(html_path.read_text(errors="ignore"))
                if item7:
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(item7, encoding="utf-8")
                filing_rows.append({**f, "cik": cik, "ticker": t, "html_path": str(html_path),
                                    "item7_path": str(text_path) if item7 else None,
                                    "item7_chars": len(item7) if item7 else 0, "run_date": run_date})
                record("10k_document", t, "ok" if item7 else "item7_not_found", started=started)
            except Exception as exc:
                record("10k_document", t, "failed", error=str(exc), started=started)

    load_duckdb(args.db, run_date, companies, fact_rows, filing_rows, log)
    failed = sum(1 for r in log if r["status"] == "failed")
    print(f"\nRun {run_id} finished: {len(fact_rows):,} facts, {len(filing_rows)} 10-Ks, {failed} failures")


def load_duckdb(db_path, run_date, companies, fact_rows, filing_rows, log):
    import duckdb
    import pandas as pd

    con = duckdb.connect(db_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    def replace_run(table, df):
        """Replace today's rows so a re-run never duplicates data (idempotent)."""
        if df.empty:
            return
        con.register("incoming", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS raw.{table} AS SELECT * FROM incoming LIMIT 0")
        con.execute(f"DELETE FROM raw.{table} WHERE run_date = ?", [run_date])
        con.execute(f"INSERT INTO raw.{table} BY NAME SELECT * FROM incoming")
        con.unregister("incoming")

    comp = pd.DataFrame(companies)
    comp["run_date"] = run_date
    replace_run("companies", comp)
    replace_run("xbrl_facts", pd.DataFrame(fact_rows))
    replace_run("filings_10k", pd.DataFrame(filing_rows))

    runs = pd.DataFrame(log)
    if not runs.empty:
        con.register("incoming", runs)
        con.execute("CREATE TABLE IF NOT EXISTS raw.pipeline_runs AS SELECT * FROM incoming LIMIT 0")
        con.execute("INSERT INTO raw.pipeline_runs BY NAME SELECT * FROM incoming")  # append-only history
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FinTrust AI SEC ingestion")
    p.add_argument("--companies", default="companies.csv")
    p.add_argument("--starter-only", action="store_true", help="only rows with starter=yes")
    p.add_argument("--years", type=int, default=3, help="how many recent 10-Ks per company")
    p.add_argument("--skip-filings", action="store_true", help="skip 10-K downloads")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--db", default="fintrust.duckdb")
    run(p.parse_args())
