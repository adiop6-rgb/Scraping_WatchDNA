import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUT_DIR = PROJECT_DIR / "out"
LOG_DIR = OUT_DIR / "logs"
PROCESSED_DIR = INPUT_DIR / "processed"
SCRAPER = PROJECT_DIR / "ai_watch_scraper.py"

URL_CANDIDATES = ["url", "URL", "product_url", "Product URL", "link", "Link"]

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def latest_csv(input_dir: Path) -> Path | None:
    files = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def is_probably_url(s: str) -> bool:
    # quick checks (not perfect, but practical)
    if not s or " " in s:
        return False
    # allow domain-only or full URL; we'll normalize later
    return bool(re.search(r"\.[a-zA-Z]{2,}(/|$)", s)) or s.startswith(("http://", "https://"))

def normalize_url(u: str) -> str | None:
    u = (u or "").strip()
    if not u:
        return None

    # If user pasted domain without scheme, add https://
    if not u.startswith(("http://", "https://")) and is_probably_url(u):
        u = "https://" + u

    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    return u

def pick_url_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("CSV has no header row / fieldnames detected.")
    # exact match first
    for c in URL_CANDIDATES:
        if c in fieldnames:
            return c
    # fallback: case-insensitive match
    lower_map = {f.lower(): f for f in fieldnames}
    for c in URL_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"Could not find a URL column. Found columns: {fieldnames}")

def read_urls(csv_path: Path) -> tuple[list[str], list[str]]:
    """
    Returns: (urls, rejected_rows_debug)
    """
    urls: list[str] = []
    rejected: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        url_col = pick_url_column(reader.fieldnames)

        for i, row in enumerate(reader, start=2):  # start=2 to account for header row = 1
            raw = (row.get(url_col) or "").strip()
            if not raw:
                continue

            norm = normalize_url(raw)
            if norm:
                urls.append(norm)
            else:
                rejected.append(f"line {i}: {raw}")

    # de-dupe while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped, rejected

def build_cmd(urls: list[str], output_path: Path) -> list[str]:
    # Adjust flags if your scraper differs
    cmd = [sys.executable, str(SCRAPER)]
    for u in urls:
        cmd += ["--product-url", u]
    cmd += ["--output", str(output_path)]
    return cmd

def run_scraper(cmd: list[str], log_path: Path) -> int:
    # write both stdout + stderr to log
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[{datetime.now().isoformat(timespec='seconds')}] Running:\n")
        logf.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, text=True, stdout=logf, stderr=logf)
        logf.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Exit code: {proc.returncode}\n")
        return proc.returncode

def move_to_processed(csv_path: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / csv_path.name
    # avoid overwrite
    if dest.exists():
        dest = PROCESSED_DIR / f"{csv_path.stem}_{now_stamp()}{csv_path.suffix}"
    shutil.move(str(csv_path), str(dest))
    return dest

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: allow passing a csv path: python run_from_csv.py input/myfile.csv
    csv_path: Path | None = None
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).expanduser().resolve()
        if not csv_path.exists() or csv_path.suffix.lower() != ".csv":
            print(f"Provided path is not a CSV that exists: {csv_path}")
            return 2
    else:
        csv_path = latest_csv(INPUT_DIR)

    if not csv_path:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] No CSV found in {INPUT_DIR}. Nothing to run.")
        return 0

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Using input CSV: {csv_path.name}")

    urls, rejected = read_urls(csv_path)
    if rejected:
        print(f"⚠️ Rejected {len(rejected)} URL-like values (invalid format). See log/urls file for details.")

    if not urls:
        print("No valid URLs found in CSV. Exiting.")
        return 0

    stamp = now_stamp()
    output_path = OUT_DIR / f"shopify_output_{stamp}.csv"
    log_path = LOG_DIR / f"run_{stamp}.log"
    urls_used_path = OUT_DIR / f"urls_used_{stamp}.txt"
    rejected_path = OUT_DIR / f"urls_rejected_{stamp}.txt"

    # Save transparency files
    urls_used_path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    if rejected:
        rejected_path.write_text("\n".join(rejected) + "\n", encoding="utf-8")

    print(f"Found {len(urls)} valid URLs.")
    print(f"Output will be: {output_path.name}")
    print(f"Log will be: {log_path.name}")

    cmd = build_cmd(urls, output_path)
    rc = run_scraper(cmd, log_path)

    if rc == 0:
        moved = move_to_processed(csv_path)
        print(f"✅ Done. Output saved to: {output_path}")
        print(f"📦 Input CSV moved to: {moved}")
        print(f"🧾 Log saved to: {log_path}")
    else:
        print(f"❌ Scraper failed with exit code {rc}")
        print(f"🧾 Check log for details: {log_path}")
        print("Input CSV was NOT moved so you can fix and re-run.")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())