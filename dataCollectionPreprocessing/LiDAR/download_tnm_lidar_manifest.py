#!/usr/bin/env python3
"""Download TNM LiDAR LAZ files from a manifest with polite parallelism.

Run this after ``build_tnm_lidar_download_manifest.py``. It downloads queued
manifest rows with bounded global, per-host, and per-project concurrency.
Existing complete files are skipped before network requests. By default the
queue is scheduled city-by-city so downstream coverage checks can become useful
as soon as a city finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "tnm_lidar_download_manifest.csv"
DEFAULT_STATUS_JSON = HERE / "tnm_lidar_download_status.json"
DEFAULT_LIDAR_ROOT = Path(r"E:\LiDAR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--lidar-root", type=Path, default=DEFAULT_LIDAR_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Global concurrent download limit.")
    parser.add_argument("--per-host", type=int, default=3)
    parser.add_argument("--per-project", type=int, default=2)
    parser.add_argument("--per-city", type=int, default=2)
    parser.add_argument(
        "--schedule",
        choices=("city", "global"),
        default="city",
        help="city drains one city before starting the next; global uses manifest priority only.",
    )
    parser.add_argument(
        "--city-token",
        action="append",
        default=[],
        help="Restrict downloads to one city token, city code, or city name. Repeat for multiple cities.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10, help="Rewrite manifest after this many completed jobs.")
    parser.add_argument("--max-downloads", type=int, default=0, help="0 means all queued rows.")
    parser.add_argument(
        "--max-city-sequential-failures",
        type=int,
        default=5,
        help="In city scheduling mode, stop launching new downloads for a city after this many row failures in a row. 0 disables.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-partial", action="store_true", help="Try HTTP Range resume for .partial files.")
    parser.add_argument(
        "--strict-manifest-size",
        action="store_true",
        help="Fail downloads whose final file size differs from manifest size_bytes, even if the server Content-Length matches.",
    )
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ssl_context(no_verify_tls: bool) -> ssl.SSLContext | None:
    if not no_verify_tls:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def read_manifest(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    needed = [
        "download_started_at",
        "download_finished_at",
        "download_attempts",
        "downloaded_bytes",
        "last_error",
    ]
    for field in needed:
        if field not in fieldnames:
            fieldnames.append(field)
        for row in rows:
            row.setdefault(field, "")
    return rows, fieldnames


def write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def destination_for(row: dict, lidar_root: Path) -> Path:
    relative = row.get("relative_path") or str(Path(row.get("canonical_city_code", "UNK")) / row["filename"])
    return lidar_root / relative


def expected_size(row: dict) -> int | None:
    value = str(row.get("size_bytes", "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def complete_existing(path: Path, size: int | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if size is not None and path.stat().st_size != size:
        return False
    return True


def complete_existing_for_row(row: dict, path: Path, size: int | None) -> bool:
    if complete_existing(path, size):
        return True
    if not path.exists() or path.stat().st_size == 0:
        return False
    if row.get("status") not in {"downloaded", "skipped_existing"}:
        return False
    try:
        recorded_bytes = int(str(row.get("downloaded_bytes", "")).strip())
    except ValueError:
        return False
    return recorded_bytes == path.stat().st_size


def normalize_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def priority_rank(row: dict) -> int:
    try:
        return int(row.get("priority_rank") or 999999999)
    except ValueError:
        return 999999999


def city_code(row: dict) -> str:
    return (row.get("canonical_city_code") or row.get("canonical_city_token") or "UNK").strip() or "UNK"


def city_filter_matches(row: dict, filters: set[str]) -> bool:
    if not filters:
        return True
    values = {
        row.get("canonical_city_code", ""),
        row.get("canonical_city_token", ""),
        row.get("canonical_city_name", ""),
    }
    for field in ("city_codes", "city_tokens", "city_names"):
        values.update(part.strip() for part in str(row.get(field, "")).split(";") if part.strip())
    return bool({normalize_token(value) for value in values} & filters)


def group_by_city(queue: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for row in sorted(queue, key=priority_rank):
        grouped.setdefault(city_code(row), []).append(row)
    return sorted(grouped.items(), key=lambda item: min(priority_rank(row) for row in item[1]))


def retry_delay(base: float, attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(base, float(retry_after))
        except ValueError:
            pass
    return base * (2 ** max(0, attempt - 1))


def download_row(row: dict, args: argparse.Namespace, context: ssl.SSLContext | None) -> tuple[str, str, int]:
    destination = destination_for(row, args.lidar_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = expected_size(row)

    if complete_existing(destination, size) and not args.overwrite:
        return "skipped_existing", "", destination.stat().st_size

    partial = destination.with_suffix(destination.suffix + ".partial")
    if args.overwrite:
        partial.unlink(missing_ok=True)
    url = row["url"]
    downloaded = 0

    for attempt in range(1, args.retries + 1):
        headers = {"User-Agent": "remoteUrbanTreeId polite TNM LiDAR downloader"}
        mode = "wb"
        resume_from = partial.stat().st_size if args.resume_partial and partial.exists() else 0
        server_expected_total = None
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=args.timeout, context=context) as response:
                if resume_from > 0 and getattr(response, "status", None) != 206:
                    partial.unlink(missing_ok=True)
                    mode = "wb"
                    resume_from = 0
                expected = size
                content_length = response.headers.get("Content-Length")
                if content_length:
                    server_expected_total = resume_from + int(content_length)
                if expected is None and content_length and resume_from == 0:
                    expected = int(content_length)
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=args.chunk_mb * 1024 * 1024)
            downloaded = partial.stat().st_size
            if size is not None and downloaded != size:
                if (
                    args.strict_manifest_size
                    or server_expected_total is None
                    or downloaded != server_expected_total
                ):
                    raise RuntimeError(f"size mismatch: expected {size}, got {downloaded}")
                partial.replace(destination)
                warning = (
                    f"manifest size_bytes mismatch accepted: manifest={size}, "
                    f"server_content_length_total={server_expected_total}, downloaded={downloaded}"
                )
                return "downloaded", warning, destination.stat().st_size
            partial.replace(destination)
            return "downloaded", "", destination.stat().st_size
        except urllib.error.HTTPError as error:
            message = f"HTTP {error.code}: {error.reason}"
            if attempt >= args.retries:
                partial.unlink(missing_ok=True)
                return "failed", message, downloaded
            time.sleep(retry_delay(args.retry_sleep, attempt, error.headers.get("Retry-After")))
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            message = str(error)
            if attempt >= args.retries:
                partial.unlink(missing_ok=True)
                return "failed", message, downloaded
            time.sleep(retry_delay(args.retry_sleep, attempt))
    return "failed", "unknown download failure", downloaded


def queued_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    queue = []
    city_filters = {normalize_token(value) for value in args.city_token if normalize_token(value)}
    for row in rows:
        if not city_filter_matches(row, city_filters):
            continue
        destination = destination_for(row, args.lidar_root)
        row["local_path"] = str(destination)
        if complete_existing_for_row(row, destination, expected_size(row)) and not args.overwrite:
            row["status"] = "skipped_existing"
            row["downloaded_bytes"] = str(destination.stat().st_size)
            continue
        if row.get("status") == "downloaded" and not args.overwrite:
            if complete_existing_for_row(row, destination, expected_size(row)):
                continue
        queue.append(row)
    if args.schedule == "global":
        queue.sort(key=priority_rank)
    else:
        queue = [row for _, city_rows in group_by_city(queue) for row in city_rows]
    if args.max_downloads:
        queue = queue[: args.max_downloads]
    return queue


def write_status(path: Path, rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts": counts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_download_queue(
    queue: list[dict],
    rows: list[dict],
    fieldnames: list[str],
    args: argparse.Namespace,
    context: ssl.SSLContext | None,
    label: str = "",
) -> dict[str, int]:
    host_semaphores: dict[str, threading.Semaphore] = {}
    project_semaphores: dict[str, threading.Semaphore] = {}
    city_semaphores: dict[str, threading.Semaphore] = {}
    lock = threading.Lock()
    completed = 0
    counts = {"downloaded": 0, "skipped_existing": 0, "failed": 0, "deferred_city_failure_threshold": 0}
    sequential_failures = 0
    stop_launching = False

    def semaphore_for(mapping: dict[str, threading.Semaphore], key: str, limit: int) -> threading.Semaphore:
        with lock:
            if key not in mapping:
                mapping[key] = threading.Semaphore(max(1, limit))
            return mapping[key]

    def worker(row: dict) -> tuple[dict, str, str, int]:
        host_sem = semaphore_for(host_semaphores, row.get("host") or "unknown", args.per_host)
        project_sem = semaphore_for(project_semaphores, row.get("project") or "unknown", args.per_project)
        city_limit = args.workers if args.schedule == "city" else args.per_city
        city_sem = semaphore_for(city_semaphores, city_code(row), city_limit)
        with host_sem, project_sem, city_sem:
            with lock:
                row["status"] = "downloading"
                row["download_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                row["download_attempts"] = str(int(row.get("download_attempts") or 0) + 1)
            status, error, bytes_written = download_row(row, args, context)
            return row, status, error, bytes_written

    def defer_unstarted(row: dict, message: str) -> None:
        row["status"] = "deferred_city_failure_threshold"
        row["last_error"] = message
        row["download_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        next_to_submit = 0
        futures = {}
        pending = set()

        def submit_more() -> None:
            nonlocal next_to_submit
            while not stop_launching and next_to_submit < len(queue) and len(pending) < args.workers:
                row = queue[next_to_submit]
                next_to_submit += 1
                future = executor.submit(worker, row)
                futures[future] = row
                pending.add(future)

        submit_more()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row, status, error, bytes_written = future.result()
                completed += 1
                counts[status] = counts.get(status, 0) + 1
                if status == "failed":
                    sequential_failures += 1
                    print(
                        f"{label}ERROR {city_code(row)} {row.get('filename')}: {error or 'download failed'}",
                        file=sys.stderr,
                    )
                else:
                    sequential_failures = 0
                    if error:
                        print(
                            f"{label}WARNING {city_code(row)} {row.get('filename')}: {error}",
                            file=sys.stderr,
                        )
                with lock:
                    row["status"] = status
                    row["last_error"] = error
                    row["downloaded_bytes"] = str(bytes_written)
                    row["download_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    if completed % max(1, args.save_every) == 0:
                        write_manifest(args.manifest, rows, fieldnames)
                        write_status(args.status_json, rows)
                print(
                    f"{label}[{completed:,}/{len(queue):,}] {status} "
                    f"{city_code(row)} {row.get('filename')}"
                )
                if (
                    args.schedule == "city"
                    and args.max_city_sequential_failures > 0
                    and sequential_failures >= args.max_city_sequential_failures
                    and not stop_launching
                ):
                    stop_launching = True
                    remaining_unstarted = queue[next_to_submit:]
                    message = (
                        f"City download deferred after {sequential_failures} sequential failed "
                        f"download(s); last_error={error or 'download failed'}"
                    )
                    for deferred_row in remaining_unstarted:
                        defer_unstarted(deferred_row, message)
                    counts["deferred_city_failure_threshold"] = counts.get("deferred_city_failure_threshold", 0) + len(remaining_unstarted)
                    completed += len(remaining_unstarted)
                    print(
                        f"{label}STOP CITY {city_code(row)}: {message}. "
                        f"Deferred {len(remaining_unstarted):,} unstarted file(s) and will move to the next city.",
                        file=sys.stderr,
                    )
                    for pending_future in list(pending):
                        if pending_future.cancel():
                            deferred_row = futures[pending_future]
                            defer_unstarted(deferred_row, message)
                            counts["deferred_city_failure_threshold"] = counts.get(
                                "deferred_city_failure_threshold", 0
                            ) + 1
                            completed += 1
                            pending.remove(pending_future)
                    write_manifest(args.manifest, rows, fieldnames)
                    write_status(args.status_json, rows)
            submit_more()

    write_manifest(args.manifest, rows, fieldnames)
    write_status(args.status_json, rows)
    return counts


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    rows, fieldnames = read_manifest(args.manifest)
    queue = queued_rows(rows, args)
    print(f"Manifest rows={len(rows):,}; queued for transfer={len(queue):,}; schedule={args.schedule}")
    if args.city_token:
        print(f"City filter: {', '.join(args.city_token)}")
    if not queue:
        write_manifest(args.manifest, rows, fieldnames)
        write_status(args.status_json, rows)
        print("Nothing to download.")
        return 0

    if args.dry_run:
        shown = 0
        groups = group_by_city(queue) if args.schedule == "city" else [("GLOBAL", queue)]
        for group_city, city_rows in groups:
            if shown >= 20:
                break
            print(f"  CITY {group_city}: {len(city_rows):,} queued")
            for row in city_rows[: max(0, 20 - shown)]:
                print(f"    {row.get('priority_rank')} {city_code(row)} {row['project']} {row['filename']}")
                shown += 1
                if shown >= 20:
                    break
        if len(queue) > shown:
            print(f"  ... {len(queue) - shown:,} more")
        return 0

    context = ssl_context(args.no_verify_tls)
    total_counts = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    if args.schedule == "city":
        city_groups = group_by_city(queue)
        for index, (group_city, city_rows) in enumerate(city_groups, start=1):
            print(f"CITY {index:,}/{len(city_groups):,} {group_city}: {len(city_rows):,} queued")
            counts = run_download_queue(city_rows, rows, fieldnames, args, context, label=f"{group_city} ")
            for status, count in counts.items():
                total_counts[status] = total_counts.get(status, 0) + count
            print(f"CITY {group_city} complete: {json.dumps(counts, sort_keys=True)}")
    else:
        total_counts = run_download_queue(queue, rows, fieldnames, args, context)

    print(json.dumps(total_counts, indent=2))
    return 1 if total_counts.get("failed", 0) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; manifest contains last periodic checkpoint.", file=sys.stderr)
        raise
