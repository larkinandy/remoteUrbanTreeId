#!/usr/bin/env python3
"""Download paired NAIP ZIPs from the NRCS Box archive.

This script intentionally follows only the archive shape described by the
current workflow:

    NAIP root -> year -> state -> xx_n and xx_c -> ZIP files

For each target county in ``naip_county_manifest.json``, it selects the
same-year natural-color and companion/CIR ZIP pair closest to 2022, then
downloads the raw ZIPs into city-code folders under ``E:/NAIP_PAIRED``:

    E:/NAIP_PAIRED/ATL/ortho_1-1_hn_s_ga063_2023_1.zip
    E:/NAIP_PAIRED/ATL/ortho_1-1_hc_s_ga063_2023_1.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BOX_ROOT = "https://nrcs.app.box.com/v/naip/folder/17936490251"
DEFAULT_MANIFEST = HERE / "naip_county_manifest.json"
DEFAULT_OUTPUT = Path(r"E:\NAIP_PAIRED")
SKIPPED_CITY_CODES = {"BAL"}

STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island",
    "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
    "DC": "district of columbia",
}

ZIP_RE = re.compile(
    r"^(?P<name>ortho_1-1_(?P<product>[a-z]{2})_s_"
    r"(?P<state>[a-z]{2})(?P<county>\d{3})_.*?"
    r"(?P<year>\d{4})_(?P<version>\d+)\.zip)$",
    re.I,
)
ZIP_NAME_RE = re.compile(r"ortho_1-1_[a-z]{2}_s_[a-z]{2}\d{3}_.*?\d{4}_\d+\.zip", re.I)
BOX_LINK_RE = re.compile(r"/v/naip/(?:folder|file)/\d+", re.I)


@dataclass(frozen=True)
class CountyInfo:
    state: str
    county_fips: str
    county_name: str
    geoid: str
    cities: tuple[dict, ...]


@dataclass(frozen=True)
class ArchiveFile:
    state: str
    county_fips: str
    year: int
    product: str
    kind: str
    name: str
    href: str


@dataclass(frozen=True)
class DownloadTask:
    city_code: str
    item: ArchiveFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--box-root-url", default=DEFAULT_BOX_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-year", type=int, default=2022)
    parser.add_argument("--year-window", type=int, default=6)
    parser.add_argument("--city-code", action="append", default=[], help="Restrict to one or more city codes.")
    parser.add_argument("--test-city", default="", help="Restrict to one city by manifest code or city name.")
    parser.add_argument("--state", action="append", default=[], help="Restrict to one or more state abbreviations.")
    parser.add_argument("--natural-product", action="append", default=["hn", "nc"])
    parser.add_argument("--companion-product", action="append", default=["hc"])
    parser.add_argument("--index-cache", type=Path, default=HERE / "naip_paired_box_index.json")
    parser.add_argument("--selection-output", type=Path, default=HERE / "naip_paired_selection.json")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--scroll-rounds", type=int, default=80)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-minutes", type=float, default=20)
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--download-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_href(href: str) -> str:
    match = BOX_LINK_RE.search(href or "")
    return match.group(0) if match else (href or "")


def box_url(href: str) -> str:
    return href if href.startswith("http") else "https://nrcs.app.box.com" + href


def with_name_sort(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("sortColumn", "name")
    query.setdefault("sortDirection", "ASC")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )


def zip_name_from_text(text: str) -> str:
    match = ZIP_NAME_RE.search(text or "")
    return match.group(0) if match else ""


def product_codes(values: list[str]) -> set[str]:
    codes = set()
    for value in values:
        code = value.strip().lower()
        codes.add(f"h{code}" if len(code) == 1 else code)
    return codes


def resolve_test_city(manifest: dict, query: str) -> str:
    wanted = normalize_token(query)
    matches = []
    for city in manifest.get("cities", []):
        code = str(city.get("code", "")).upper()
        if wanted in {normalize_token(code), normalize_token(city.get("city", ""))}:
            matches.append((code, city.get("city", "")))
    if not matches:
        raise SystemExit(f"--test-city did not match a manifest city: {query!r}")
    if len(matches) > 1:
        names = ", ".join(f"{code} ({name})" for code, name in matches)
        raise SystemExit(f"--test-city matched multiple cities: {names}")
    return matches[0][0]


def target_counties(manifest: dict, city_codes: set[str], states: set[str]) -> dict[tuple[str, str], CountyInfo]:
    counties: dict[tuple[str, str], dict] = {}
    for city in manifest.get("cities", []):
        code = str(city.get("code", "")).upper()
        if code in SKIPPED_CITY_CODES:
            continue
        if city_codes and code not in city_codes:
            continue
        for county in city.get("counties", []):
            state = str(county.get("state", "")).upper()
            county_fips = str(county.get("county_fips", "")).zfill(3)
            if states and state not in states:
                continue
            key = (state, county_fips)
            entry = counties.setdefault(
                key,
                {
                    "state": state,
                    "county_fips": county_fips,
                    "county_name": county.get("name", ""),
                    "geoid": county.get("geoid") or f"{state}{county_fips}",
                    "cities": [],
                },
            )
            entry["cities"].append({"city": city.get("city", ""), "code": code})
    return {key: CountyInfo(**{**value, "cities": tuple(value["cities"])}) for key, value in counties.items()}


def year_key(year: int, target_year: int) -> tuple[int, int, int]:
    return (abs(year - target_year), 0 if year >= target_year else 1, year if year >= target_year else -year)


def parse_year_label(label: str) -> int | None:
    label = label.strip()
    return int(label) if re.fullmatch(r"(19|20)\d{2}", label) else None


def parse_archive_file(name: str, href: str, natural: set[str], companion: set[str]) -> ArchiveFile | None:
    match = ZIP_RE.match(name)
    if not match:
        return None
    product = match.group("product").lower()
    if product in natural:
        kind = "natural"
    elif product in companion:
        kind = "companion"
    else:
        return None
    return ArchiveFile(
        state=match.group("state").upper(),
        county_fips=match.group("county"),
        year=int(match.group("year")),
        product=product,
        kind=kind,
        name=match.group("name"),
        href=normalize_href(href),
    )


def browser_import():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Install Playwright for the Python interpreter running this script:\n"
            f"  {sys.executable} -m pip install playwright",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return sync_playwright


def visible_folder_rows(page) -> list[dict[str, str]]:
    return page.evaluate(
        """
        () => Array.from(document.querySelectorAll('a[href*="/v/naip/folder/"]')).map(a => ({
            text: (a.innerText || a.textContent || a.getAttribute('aria-label') || a.getAttribute('title') || '')
                .replace(/\\s+/g, ' ')
                .trim(),
            href: a.getAttribute('href') || ''
        })).filter(row => row.text && row.href)
        """
    )


def visible_file_rows(page) -> list[dict[str, str]]:
    return page.evaluate(
        """
        () => {
            const clean = value => (value || '').replace(/\\s+/g, ' ').trim();
            const rows = [];
            for (const a of Array.from(document.querySelectorAll('a[href*="/v/naip/file/"]'))) {
                const record = a.closest('[role="row"], li, tr, [data-testid]') || a;
                rows.push({text: clean(record.innerText || record.textContent || a.innerText || ''), href: a.getAttribute('href') || ''});
            }
            return rows;
        }
        """
    )


def scroll_step(page) -> int:
    return page.evaluate(
        """
        () => {
            const scrollables = Array.from(document.querySelectorAll('*'))
                .filter(e => e.scrollHeight > e.clientHeight + 50)
                .map(e => {
                    const text = (e.innerText || e.textContent || '').trim();
                    const cls = String(e.className || '');
                    const priority =
                        cls.includes('ItemListLayout') ? 0 :
                        e.getAttribute('role') === 'grid' ? 1 :
                        text ? 2 : 3;
                    return {e, priority, range: e.scrollHeight - e.clientHeight};
                })
                .sort((a, b) => a.priority - b.priority || b.range - a.range);
            for (const {e} of scrollables.slice(0, 4)) {
                e.scrollTop = Math.min(e.scrollTop + Math.max(900, e.clientHeight), e.scrollHeight);
            }
            window.scrollBy(0, 1200);
            return scrollables
                .slice(0, 4)
                .reduce((total, {e}) => total + e.scrollTop + e.scrollHeight + e.clientHeight, 0);
        }
        """
    )


def collect_visible_rows(page, row_func, rounds: int, settle_seconds: float) -> list[dict[str, str]]:
    seen: dict[tuple[str, str], dict[str, str]] = {}
    last_height = -1
    stable = 0
    for _ in range(rounds):
        before = len(seen)
        for row in row_func(page):
            text = (row.get("text") or "").strip()
            href = normalize_href(row.get("href") or "")
            if text and href:
                seen[(text.lower(), href)] = {"text": text, "href": href}
        height = scroll_step(page)
        stable = stable + 1 if height == last_height and len(seen) == before else 0
        last_height = height
        if stable >= 5:
            break
        time.sleep(settle_seconds)
    for row in row_func(page):
        text = (row.get("text") or "").strip()
        href = normalize_href(row.get("href") or "")
        if text and href:
            seen[(text.lower(), href)] = {"text": text, "href": href}
    return list(seen.values())


def folder_links(page, url: str, args: argparse.Namespace) -> dict[str, str]:
    print(f"Folders: {url}")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    links: dict[str, str] = {}
    for row in collect_visible_rows(page, visible_folder_rows, args.scroll_rounds, args.settle_seconds):
        links.setdefault(row["text"].strip().lower(), normalize_href(row["href"]))
    print(f"  found {len(links)} folder link(s)")
    return links


def archive_files(page, url: str, args: argparse.Namespace, natural: set[str], companion: set[str]) -> list[ArchiveFile]:
    files: dict[str, ArchiveFile] = {}
    visited: set[str] = set()
    next_url = with_name_sort(url)
    while next_url and next_url not in visited:
        visited.add(next_url)
        print(f"Files:   {next_url}")
        page.goto(next_url, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        for row in visible_file_rows(page):
            name = zip_name_from_text(row["text"])
            if not name:
                continue
            item = parse_archive_file(name, row["href"], natural, companion)
            if item:
                files[item.name.lower()] = item
        next_href = page.evaluate(
            """
            () => {
                const link = document.querySelector('a[aria-label="Next Page"][href]');
                return link ? link.getAttribute('href') : '';
            }
            """
        )
        next_url = box_url(next_href) if next_href else ""
    print(f"  found {len(files)} ZIP file link(s) across {len(visited)} page(s)")
    return sorted(files.values(), key=lambda item: (item.state, item.county_fips, item.year, item.kind, item.name))


def find_state_folder(links: dict[str, str], state: str) -> str:
    for label in [state.lower(), STATE_NAMES.get(state, "").lower()]:
        if label and label in links:
            return links[label]
    return ""


def crawl_index(args: argparse.Namespace, targets: dict[tuple[str, str], CountyInfo], natural: set[str], companion: set[str]) -> list[ArchiveFile]:
    sync_playwright = browser_import()
    wanted_states = sorted({state for state, _county in targets})
    wanted_counties = set(targets)
    found: dict[str, ArchiveFile] = {}
    timeout_ms = int(args.timeout_minutes * 60_000)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=not args.headed)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        root = folder_links(page, args.box_root_url, args)
        year_folders = []
        for label, href in root.items():
            year = parse_year_label(label)
            if year is not None and abs(year - args.target_year) <= args.year_window:
                year_folders.append((year_key(year, args.target_year), year, href))
        year_folders.sort(key=lambda row: row[0])
        discovered_years = [year for _key, year, _href in year_folders]
        print("Year order: " + (", ".join(str(year) for year in discovered_years) or "(none)"))
        if args.target_year not in discovered_years:
            print(
                f"WARNING: target year {args.target_year} was not found in the root folder listing. "
                "If it exists in Box, increase --scroll-rounds or run with --headed to inspect the page."
            )

        for _key, year, year_href in year_folders:
            year_links = folder_links(page, box_url(year_href), args)
            for state in wanted_states:
                state_href = find_state_folder(year_links, state)
                if not state_href:
                    print(f"  {year}: no {state} folder")
                    continue
                state_links = folder_links(page, box_url(state_href), args)
                for suffix, kind in [("n", "natural"), ("c", "companion")]:
                    product_href = state_links.get(f"{state.lower()}_{suffix}", "")
                    if not product_href:
                        print(f"  {year}: no {state.lower()}_{suffix} folder")
                        continue
                    for item in archive_files(page, box_url(product_href), args, natural, companion):
                        if (item.state, item.county_fips) in wanted_counties:
                            found[item.name.lower()] = item
                            print(f"  found {kind}: {item.name}")
            if set(select_pairs(list(found.values()), targets, args.target_year)) >= wanted_counties:
                print("All target counties have a complete pair.")
                break
        browser.close()

    return sorted(found.values(), key=lambda item: (item.state, item.county_fips, item.year, item.kind, item.name))


def crawl_index_and_download_matches(
    args: argparse.Namespace,
    targets: dict[tuple[str, str], CountyInfo],
    natural: set[str],
    companion: set[str],
) -> tuple[list[ArchiveFile], int, list[dict]]:
    sync_playwright = browser_import()
    wanted_counties = set(targets)
    found: dict[str, ArchiveFile] = {}
    downloaded_counties: set[tuple[str, str]] = set()
    downloaded_this_run = 0
    failures: list[dict] = []
    timeout_ms = int(args.timeout_minutes * 60_000)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=not args.headed)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        root = folder_links(page, args.box_root_url, args)
        year_folders = []
        for label, href in root.items():
            year = parse_year_label(label)
            if year is not None and abs(year - args.target_year) <= args.year_window:
                year_folders.append((year_key(year, args.target_year), year, href))
        year_folders.sort(key=lambda row: row[0])
        discovered_years = [year for _key, year, _href in year_folders]
        print("Year order: " + (", ".join(str(year) for year in discovered_years) or "(none)"))
        if args.target_year not in discovered_years:
            print(
                f"WARNING: target year {args.target_year} was not found in the root folder listing. "
                "If it exists in Box, increase --scroll-rounds or run with --headed to inspect the page."
            )

        for _key, year, year_href in year_folders:
            remaining_counties = wanted_counties - downloaded_counties
            if not remaining_counties:
                print("All target counties have been downloaded or already existed locally.")
                break
            wanted_states = sorted({state for state, _county in remaining_counties})
            year_links = folder_links(page, box_url(year_href), args)
            for state in wanted_states:
                state_href = find_state_folder(year_links, state)
                if not state_href:
                    print(f"  {year}: no {state} folder")
                    continue
                state_links = folder_links(page, box_url(state_href), args)
                for suffix, kind in [("n", "natural"), ("c", "companion")]:
                    product_href = state_links.get(f"{state.lower()}_{suffix}", "")
                    if not product_href:
                        print(f"  {year}: no {state.lower()}_{suffix} folder")
                        continue
                    for item in archive_files(page, box_url(product_href), args, natural, companion):
                        if (item.state, item.county_fips) in remaining_counties:
                            found[item.name.lower()] = item
                            print(f"  found {kind}: {item.name}")

                    selected_now = select_pairs(list(found.values()), targets, args.target_year)
                    ready = sorted(set(selected_now) - downloaded_counties)
                    for key in ready:
                        pair = selected_now[key]
                        print(f"Complete pair ready: {key[0]}{key[1]} {pair['natural'].year}; downloading now.")
                        for task in download_tasks({key: pair}, targets):
                            try:
                                downloaded_this_run += int(download_one(page, task, args.output, args.retries, timeout_ms))
                            except KeyboardInterrupt:
                                browser.close()
                                raise
                            except Exception as exc:
                                failures.append({"city_code": task.city_code, "file": task.item.name, "error": str(exc)})
                                print(f"FAILED {task.city_code} {task.item.name}: {exc}", file=sys.stderr)
                        downloaded_counties.add(key)
            if downloaded_counties >= wanted_counties:
                print("All target counties have complete pairs handled.")
                break
        browser.close()

    return (
        sorted(found.values(), key=lambda item: (item.state, item.county_fips, item.year, item.kind, item.name)),
        downloaded_this_run,
        failures,
    )


def cached_index_covers(path: Path, targets: dict[tuple[str, str], CountyInfo]) -> bool:
    if not path.exists():
        return False
    data = load_json(path)
    required = {"state", "county_fips", "year", "product", "kind", "name", "href"}
    if any(not required <= set(row) for row in data.get("files", [])):
        print(f"Ignoring stale index with old schema: {path}")
        return False
    cached = {(row["state"], str(row["county_fips"]).zfill(3)) for row in data.get("files", [])}
    return cached >= set(targets)


def load_or_crawl_index(args: argparse.Namespace, targets: dict[tuple[str, str], CountyInfo], natural: set[str], companion: set[str]) -> list[ArchiveFile]:
    if args.index_cache.exists() and not args.refresh_index and cached_index_covers(args.index_cache, targets):
        print(f"Using cached index: {args.index_cache}")
        rows = load_json(args.index_cache).get("files", [])
        return [ArchiveFile(**{**row, "county_fips": str(row["county_fips"]).zfill(3), "year": int(row["year"])}) for row in rows]

    files = crawl_index(args, targets, natural, companion)
    args.index_cache.write_text(
        json.dumps(
            {
                "box_root_url": args.box_root_url,
                "target_year": args.target_year,
                "target_counties": [asdict(county) for county in targets.values()],
                "files": [asdict(item) for item in files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote index: {args.index_cache}")
    return files


def select_pairs(files: list[ArchiveFile], targets: dict[tuple[str, str], CountyInfo], target_year: int) -> dict[tuple[str, str], dict[str, ArchiveFile]]:
    by_county_year: dict[tuple[str, str, int], dict[str, list[ArchiveFile]]] = {}
    for item in files:
        if (item.state, item.county_fips) not in targets:
            continue
        by_county_year.setdefault((item.state, item.county_fips, item.year), {}).setdefault(item.kind, []).append(item)

    selected = {}
    for state, county_fips in sorted(targets):
        candidates = []
        for (row_state, row_county, year), grouped in by_county_year.items():
            if (row_state, row_county) != (state, county_fips):
                continue
            if grouped.get("natural") and grouped.get("companion"):
                candidates.append(
                    (
                        year_key(year, target_year),
                        {
                            "natural": sorted(grouped["natural"], key=lambda row: row.name)[0],
                            "companion": sorted(grouped["companion"], key=lambda row: row.name)[0],
                        },
                    )
                )
        if candidates:
            selected[(state, county_fips)] = sorted(candidates, key=lambda row: row[0])[0][1]
    return selected


def destination(output: Path, task: DownloadTask) -> Path:
    return output / task.city_code / task.item.name


def download_one(page, task: DownloadTask, output: Path, retries: int, timeout_ms: int) -> bool:
    item = task.item
    dest = destination(output, task)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already downloaded: {dest}", flush=True)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading: {item.name} -> {dest}", flush=True)
            page.goto(box_url(item.href), wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            download = trigger_box_download(page, timeout_ms)
            partial.unlink(missing_ok=True)
            print("download started; waiting for the full ZIP to finish...", flush=True)
            download.save_as(partial)
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            partial.replace(dest)
            print(f"downloaded: {dest}", flush=True)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            print(f"attempt {attempt}/{retries} failed for {item.name}: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(min(30, attempt * 5))
    raise RuntimeError(f"failed to download {item.name}")


def trigger_box_download(page, timeout_ms: int):
    locators = [
        page.get_by_test_id("large-download-button-with-text"),
        page.get_by_role("button", name=re.compile("download", re.I)),
        page.get_by_role("link", name=re.compile("download", re.I)),
        page.locator(
            'button[aria-label*="Download"], '
            'a[aria-label*="Download"], '
            '[role="button"][aria-label*="Download"]'
        ).first,
    ]
    errors = []
    for locator in locators:
        try:
            with page.expect_download(timeout=timeout_ms) as event:
                locator.click(timeout=10_000)
            return event.value
        except Exception as exc:
            errors.append(str(exc).splitlines()[0])
    raise RuntimeError("Could not trigger Box download button. Tried selectors: " + " | ".join(errors))


def download_tasks(
    selected: dict[tuple[str, str], dict[str, ArchiveFile]],
    targets: dict[tuple[str, str], CountyInfo],
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    seen: set[tuple[str, str]] = set()
    for key, pair in sorted(selected.items()):
        county = targets[key]
        for city in county.cities:
            city_code = str(city.get("code", "")).upper()
            for item in [pair["natural"], pair["companion"]]:
                task_key = (city_code, item.name)
                if task_key in seen:
                    continue
                seen.add(task_key)
                tasks.append(DownloadTask(city_code=city_code, item=item))
    return tasks


def locally_complete_pairs(
    output: Path,
    targets: dict[tuple[str, str], CountyInfo],
    target_year: int,
    year_window: int,
    natural: set[str],
    companion: set[str],
) -> dict[tuple[str, str], dict[str, ArchiveFile]]:
    """Return target county pairs already present in every target city folder."""
    city_files: dict[str, dict[tuple[str, str, int, str], ArchiveFile]] = {}
    for county in targets.values():
        for city in county.cities:
            city_code = str(city.get("code", "")).upper()
            if city_code in city_files:
                continue
            folder = output / city_code
            by_key: dict[tuple[str, str, int, str], ArchiveFile] = {}
            if folder.exists():
                for path in folder.glob("*.zip"):
                    if not path.exists() or path.stat().st_size <= 0:
                        continue
                    item = parse_archive_file(path.name, "", natural, companion)
                    if item and abs(item.year - target_year) <= year_window:
                        by_key[(item.state, item.county_fips, item.year, item.kind)] = item
            city_files[city_code] = by_key

    complete: dict[tuple[str, str], dict[str, ArchiveFile]] = {}
    for key, county in sorted(targets.items()):
        candidate_years = set()
        for city in county.cities:
            city_code = str(city.get("code", "")).upper()
            for state, county_fips, year, kind in city_files.get(city_code, {}):
                if (state, county_fips) == key and kind == "natural":
                    candidate_years.add(year)

        year_candidates = []
        for year in candidate_years:
            natural_item = None
            companion_item = None
            all_city_files_present = True
            for city in county.cities:
                city_code = str(city.get("code", "")).upper()
                files = city_files.get(city_code, {})
                city_natural = files.get((key[0], key[1], year, "natural"))
                city_companion = files.get((key[0], key[1], year, "companion"))
                if not city_natural or not city_companion:
                    all_city_files_present = False
                    break
                natural_item = city_natural
                companion_item = city_companion
            if all_city_files_present and natural_item and companion_item:
                year_candidates.append(
                    (year_key(year, target_year), {"natural": natural_item, "companion": companion_item})
                )
        if year_candidates:
            complete[key] = sorted(year_candidates, key=lambda row: row[0])[0][1]
    return complete


def write_selection(path: Path, selected: dict[tuple[str, str], dict[str, ArchiveFile]], targets: dict[tuple[str, str], CountyInfo]) -> None:
    missing = []
    rows = []
    for key, county in sorted(targets.items()):
        pair = selected.get(key)
        if not pair:
            missing.append({**asdict(county), "reason": "no same-year natural/companion pair found"})
            continue
        rows.append(
            {
                **asdict(county),
                "selected_year": pair["natural"].year,
                "natural": asdict(pair["natural"]),
                "companion": asdict(pair["companion"]),
            }
        )
    path.write_text(json.dumps({"selected": rows, "missing": missing}, indent=2), encoding="utf-8")
    print(f"Wrote selection: {path}")


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    city_codes = {code.upper() for code in args.city_code}
    if args.test_city:
        test_code = resolve_test_city(manifest, args.test_city)
        if city_codes and city_codes != {test_code}:
            raise SystemExit("--test-city cannot be combined with a different --city-code.")
        city_codes = {test_code}
        print(f"Test-city mode: limiting run to {test_code} ({args.test_city})")
    skipped_requested = sorted(SKIPPED_CITY_CODES if not city_codes else city_codes & SKIPPED_CITY_CODES)
    if skipped_requested:
        print(
            "Skipping city code(s) with non-paired NAIP storage: "
            + ", ".join(skipped_requested)
        )
    if city_codes and city_codes <= SKIPPED_CITY_CODES:
        print("No paired NAIP downloads requested after skipped city filtering.")
        return 0

    targets = target_counties(manifest, city_codes=city_codes, states={state.upper() for state in args.state})
    if not targets:
        raise SystemExit("No target counties matched the manifest and filters.")
    print(f"Target counties: {len(targets)}")
    print(f"Target states: {', '.join(sorted({state for state, _county in targets}))}")

    natural = product_codes(args.natural_product)
    companion = product_codes(args.companion_product)
    selected = locally_complete_pairs(args.output, targets, args.target_year, args.year_window, natural, companion)
    if selected:
        print(f"Already complete locally before crawl: {len(selected)} county pair(s)")

    remaining_targets = {key: value for key, value in targets.items() if key not in selected}
    downloaded = 0
    failures: list[dict] = []
    if remaining_targets:
        print(f"County pair(s) still needing Box lookup: {len(remaining_targets)}")
        use_cached_index = (
            args.index_cache.exists()
            and not args.refresh_index
            and cached_index_covers(args.index_cache, remaining_targets)
        )
        if args.dry_run or use_cached_index:
            files = load_or_crawl_index(args, remaining_targets, natural, companion)
        else:
            files, downloaded, failures = crawl_index_and_download_matches(args, remaining_targets, natural, companion)
            args.index_cache.write_text(
                json.dumps(
                    {
                        "box_root_url": args.box_root_url,
                        "target_year": args.target_year,
                        "target_counties": [asdict(county) for county in remaining_targets.values()],
                        "files": [asdict(item) for item in files],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote index: {args.index_cache}")
        selected.update(select_pairs(files, remaining_targets, args.target_year))
    else:
        print("All requested county pairs are already downloaded locally; skipping Box crawl.")

    write_selection(args.selection_output, selected, targets)

    queue = download_tasks(selected, targets)
    if args.download_limit:
        queue = queue[: args.download_limit]

    if args.dry_run:
        print("Dry run only: remove --dry-run to download files.")
        for i, task in enumerate(queue, 1):
            dest = destination(args.output, task)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[{i}/{len(queue)}] already downloaded: {dest}")
            else:
                print(f"[{i}/{len(queue)}] would download {task.item.kind}: {dest}")
        return 0 if selected else 1

    sync_playwright = browser_import()
    args.output.mkdir(parents=True, exist_ok=True)
    timeout_ms = int(args.timeout_minutes * 60_000)
    failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=not args.headed)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        for i, task in enumerate(queue, 1):
            item = task.item
            print(f"[{i}/{len(queue)}] {task.city_code} {item.state}{item.county_fips} {item.year} {item.kind}")
            try:
                downloaded += int(download_one(page, task, args.output, args.retries, timeout_ms))
            except KeyboardInterrupt:
                print("Stopped. Rerun the same command to resume.")
                browser.close()
                return 130
            except Exception as exc:
                failures.append({"city_code": task.city_code, "file": item.name, "error": str(exc)})
        browser.close()

    summary = {
        "selected_counties": len(selected),
        "queued_files": len(queue),
        "downloaded_this_run": downloaded,
        "failures": failures,
        "selection_manifest": str(args.selection_output),
    }
    summary_path = args.output / "naip_paired_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    return 1 if failures or not selected else 0


if __name__ == "__main__":
    raise SystemExit(main())
