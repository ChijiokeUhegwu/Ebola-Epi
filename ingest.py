"""Download, clean, and normalize Ebola outbreak data from public sources.

The script queries the HDX CKAN API for package metadata, downloads the
associated CSV resources, and validates the WHO Global Health Observatory
(GHO) API for Ebola indicators. Normalized country and health-zone tables are
written to the local data directory for the Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
HDX_API = "https://data.humdata.org/api/3/action/package_show"
GHO_INDICATORS = "https://ghoapi.azureedge.net/api/Indicator"

CASE_INDICATOR = "Cumulative number of confirmed, probable and suspected Ebola cases"
DEATH_INDICATOR = "Cumulative number of confirmed, probable and suspected Ebola deaths"

PACKAGES = {
    "west_africa": {
        "package_id": "ebola-cases-2014",
        "title": "West Africa Ebola outbreak 2014-2016",
        "resource_name": "ebola_data_db_format.csv",
        "raw_name": "ebola_data_db_format.csv",
        "output_name": "ebola_west_africa.csv",
        "fallback_url": (
            "https://data.humdata.org/dataset/0d089fa0-3567-4b01-9c03-39d340ff34e3/"
            "resource/c59b5722-ca4b-41ca-a446-472d6d824d01/download/ebola_data_db_format.csv"
        ),
    },
    "drc": {
        "package_id": "republique-democratique-du-congo-cas-et-deces-d-ebola",
        "title": "DRC Ebola cases and deaths",
        "resource_name": "drc_ebola_cases_consolidated.csv",
        "raw_name": "drc_ebola_cases_consolidated.csv",
        "output_name": "ebola_drc_2026.csv",
        "fallback_url": (
            "https://data.humdata.org/dataset/e902b209-b7bc-42f9-893a-294276d7cc62/"
            "resource/f7c9b552-0d41-4559-9fae-c450175fc39c/download/drc_ebola_cases_consolidated.csv"
        ),
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    """Print a timestamped pipeline message without buffering."""
    print(f"[ingest] {message}", flush=True)


def build_session() -> requests.Session:
    """Create a requests session with retries and a descriptive user agent."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ebola-epi-dashboard/1.0 (public health research)",
            "Accept": "application/json, text/csv, */*",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_package_metadata(session: requests.Session, package_id: str) -> dict | None:
    """Return HDX package metadata, or None when the API is unavailable."""
    try:
        response = session.get(HDX_API, params={"id": package_id}, timeout=(15, 60))
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") and payload.get("result"):
            return payload["result"]
        log(f"HDX returned no metadata for {package_id}")
    except Exception as exc:
        log(f"HDX metadata request failed for {package_id}: {exc}")
    return None


def choose_resource(metadata: dict, resource_name: str) -> dict | None:
    """Prefer the expected resource name, then fall back to the first CSV."""
    for resource in metadata.get("resources", []):
        name = (resource.get("name") or "").lower()
        if resource_name.lower() in name:
            return resource
    for resource in metadata.get("resources", []):
        if (resource.get("format") or "").upper() == "CSV":
            return resource
    return None


def probe_who_gho(session: requests.Session) -> dict:
    """Check GHO for Ebola indicators and record the result in metadata."""
    try:
        response = session.get(
            GHO_INDICATORS,
            params={"$filter": "contains(IndicatorName,'Ebola')"},
            timeout=(15, 45),
        )
        response.raise_for_status()
        values = response.json().get("value", [])
        return {
            "status": "reachable",
            "ebola_indicator_count": len(values),
            "indicators": [item.get("IndicatorCode") for item in values],
            "note": "WHO GHO exposes no Ebola case/death indicator; HDX streams are used.",
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    """Download a file to a temporary path before replacing the cached file.

    Writing through a .part file prevents a partial download from being mistaken
    for a valid cache on the next run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            log(f"Downloading {url} (attempt {attempt})")
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0))
                written = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)

            # A truncated response should be retried rather than cached.
            if expected and written < expected * 0.95:
                raise IOError(f"incomplete download ({written} of {expected} bytes)")
            temporary.replace(destination)
            log(f"Saved {destination.name} ({destination.stat().st_size:,} bytes)")
            return
        except Exception as exc:
            last_error = exc
            log(f"Download attempt {attempt} failed: {exc}")
            time.sleep(2 * attempt)
    if temporary.exists():
        temporary.unlink()
    raise RuntimeError(f"Could not download {url}: {last_error}")


# ---------------------------------------------------------------------------
# Source normalization
# ---------------------------------------------------------------------------


def _daily_frame(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Align sparse report series onto a complete daily index."""
    all_dates = pd.DatetimeIndex([])
    for series in series_map.values():
        all_dates = all_dates.union(series.index)
    full_index = pd.date_range(all_dates.min(), all_dates.max(), freq="D")
    frame = pd.DataFrame(index=full_index)
    for name, series in series_map.items():
        # Interpolate between reports, then carry the first and last values outward.
        aligned = series.reindex(full_index).interpolate(method="linear", limit_area="inside")
        aligned = aligned.ffill().bfill()
        frame[name] = aligned
    return frame


def normalize_west_africa(raw_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert the long WHO GAR table into daily country-level case and death series."""
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {"Indicator", "Country", "Date", "value"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected West Africa schema: {list(raw.columns)}")

    frame = pd.DataFrame(
        {
            "indicator": raw["Indicator"].astype(str).str.strip(),
            "country": raw["Country"].astype(str).str.strip(),
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["date", "country", "value"])

    # The source includes a few "Guinea 2" / "Liberia 2" revision series.
    # They are not separate countries, so exclude them from the country curves.
    frame = frame[~frame["country"].str.contains(r"\s2$", regex=True)]
    frame = frame[frame["indicator"].isin([CASE_INDICATOR, DEATH_INDICATOR])]

    cases = (
        frame[frame["indicator"] == CASE_INDICATOR]
        .groupby(["date", "country"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "cases"})
    )
    deaths = (
        frame[frame["indicator"] == DEATH_INDICATOR]
        .groupby(["date", "country"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "deaths"})
    )
    # An outer merge preserves reports that contain cases but not deaths, or vice versa.
    merged = cases.merge(deaths, on=["date", "country"], how="outer")

    countries = []
    for country, group in merged.groupby("country"):
        group = group.sort_values("date").drop_duplicates("date", keep="last")
        date_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        series_map = {}
        for column in ["cases", "deaths"]:
            series = group.set_index("date")[column].sort_index()
            if not series.notna().any():
                series = pd.Series(0.0, index=date_index)
            aligned = series.reindex(date_index).interpolate(method="linear", limit_area="inside")
            # Cumulative totals should never fall after a reporting revision.
            aligned = aligned.ffill().bfill().clip(lower=0).cummax()
            series_map[column] = aligned
        daily = _daily_frame(series_map).reset_index().rename(columns={"index": "date"})
        daily["country"] = country
        countries.append(daily)

    normalized = pd.concat(countries, ignore_index=True)
    normalized["source"] = "WHO GAR situation reports via HDX"
    normalized = normalized[["date", "country", "cases", "deaths", "source"]]
    normalized = normalized.sort_values(["country", "date"]).reset_index(drop=True)
    normalized.to_csv(output_path, index=False)
    return normalized


def normalize_drc(raw_path: Path, output_path: Path, subnational_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build national daily series and a health-zone snapshot from DRC reports."""
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {
        "location_level",
        "reference_date",
        "measure",
        "case_classification",
        "value",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected DRC schema: {list(raw.columns)}")

    raw["reference_date"] = pd.to_datetime(raw["reference_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["reference_date", "value"])

    # location_level 0 is the national total; level 3 is a health-zone row.
    national = raw[raw["location_level"].astype(str) == "0"]

    def pick(measure: str, classification: str) -> pd.Series:
        """Select one measure/classification pair and collapse duplicate dates."""
        subset = national[
            (national["measure"] == measure)
            & (national["case_classification"] == classification)
        ]
        return subset.groupby("reference_date")["value"].max().sort_index()

    series_map = {
        "cases": pick("cases", "confirmed"),
        "deaths": pick("deaths", "confirmed"),
        "suspected_cases": pick("cases", "suspected"),
    }
    for name, series in series_map.items():
        if series.empty:
            series_map[name] = pd.Series(dtype=float)
    date_index = pd.date_range(
        min(series.index.min() for series in series_map.values() if not series.empty),
        max(series.index.max() for series in series_map.values() if not series.empty),
        freq="D",
    )
    normalized_daily = pd.DataFrame(index=date_index)
    for name, series in series_map.items():
        if series.empty:
            normalized_daily[name] = 0.0
            continue
        aligned = series.reindex(date_index).interpolate(method="linear", limit_area="inside")
        aligned = aligned.ffill().bfill().clip(lower=0)

        # Confirmed cases and deaths are cumulative; suspected cases can be reclassified.
        if name in {"cases", "deaths"}:
            aligned = aligned.cummax()
        normalized_daily[name] = aligned

    normalized = normalized_daily.reset_index().rename(columns={"index": "date", "suspected_cases": "suspected_cases"})
    normalized["country"] = "Democratic Republic of the Congo"
    normalized["source"] = "INRB-UMIE / HDX BDBV2026 sitreps"
    normalized = normalized[
        ["date", "country", "cases", "deaths", "suspected_cases", "source"]
    ].sort_values("date").reset_index(drop=True)
    normalized.to_csv(output_path, index=False)

    subnational = raw[raw["location_level"].astype(str) == "3"].copy()
    subnational["location_code"] = subnational["location_code"].fillna(
        subnational["location_name"]
    )

    # The maximum cumulative value is the latest known total for each health zone.
    case_rows = subnational[
        (subnational["measure"] == "cases")
        & (subnational["case_classification"] == "confirmed")
    ]
    death_rows = subnational[
        (subnational["measure"] == "deaths")
        & (subnational["case_classification"] == "confirmed")
    ]
    province_cases = (
        case_rows.groupby(["location_code", "location_name"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "cases"})
    )
    province_deaths = (
        death_rows.groupby(["location_code", "location_name"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "deaths"})
    )
    provinces = province_cases.merge(
        province_deaths, on=["location_code", "location_name"], how="left"
    )
    provinces["deaths"] = provinces["deaths"].fillna(0)
    provinces = provinces.sort_values("cases", ascending=False).reset_index(drop=True)
    provinces.to_csv(subnational_path, index=False)
    return normalized, provinces


# ---------------------------------------------------------------------------
# Metadata and pipeline orchestration
# ---------------------------------------------------------------------------


def describe(frame: pd.DataFrame, date_column: str = "date") -> dict:
    """Summarize row counts, coverage, and totals for ingestion metadata."""
    countries = sorted(frame["country"].dropna().unique().tolist()) if "country" in frame else []
    info = {
        "rows": int(len(frame)),
        "start": None,
        "end": None,
        "countries": countries,
        "cases": int(round(frame["cases"].max())) if "cases" in frame and len(frame) else 0,
        "deaths": int(round(frame["deaths"].max())) if "deaths" in frame and len(frame) else 0,
    }
    if date_column in frame and len(frame):
        dates = pd.to_datetime(frame[date_column])
        info["start"] = dates.min().date().isoformat()
        info["end"] = dates.max().date().isoformat()
    return info


def run(data_dir: Path, refresh: bool, offline: bool) -> dict:
    """Run the full ingestion, normalization, and metadata pipeline."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = build_session() if not offline else None
    metadata: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "who_gho_probe": {"status": "skipped"},
        "datasets": {},
    }
    if session is not None:
        metadata["who_gho_probe"] = probe_who_gho(session)
        log(f"WHO GHO probe: {metadata['who_gho_probe']['status']}")

    for key, config in PACKAGES.items():
        raw_path = raw_dir / config["raw_name"]
        resource_url = config["fallback_url"]
        package_metadata = None

        # Refresh source URLs from HDX when online; the fallback URL keeps the
        # pipeline usable if the metadata API is temporarily unavailable.
        if session is not None:
            package_metadata = fetch_package_metadata(session, config["package_id"])
            if package_metadata:
                resource = choose_resource(package_metadata, config["resource_name"])
                if resource and resource.get("url"):
                    resource_url = resource["url"]

        if refresh or not raw_path.exists():
            if session is None:
                raise FileNotFoundError(
                    f"Offline mode requires cached data at {raw_path}. Run without --offline."
                )
            try:
                download_file(session, resource_url, raw_path)
            except Exception as exc:
                if not raw_path.exists():
                    raise
                log(f"Using cached {raw_path.name} after download failure: {exc}")
        else:
            log(f"Using cached {raw_path.name}")

        # Each source has its own schema and normalization rules.
        if key == "west_africa":
            normalized = normalize_west_africa(raw_path, data_dir / config["output_name"])
            datasets = {"west_africa": normalized}
        else:
            normalized, provinces = normalize_drc(
                raw_path,
                data_dir / config["output_name"],
                data_dir / "ebola_drc_subnational.csv",
            )
            datasets = {"drc": normalized, "drc_subnational": provinces}

        for dataset_name, frame in datasets.items():
            info = describe(frame)
            info["rows"] = int(len(frame))
            if dataset_name == "drc_subnational":
                info["locations"] = int(len(frame))
            description = {
                "package_id": config["package_id"],
                "title": package_metadata.get("title") if package_metadata else config["title"],
                "organization": (
                    package_metadata.get("organization", {}).get("title")
                    if package_metadata
                    else "HDX"
                ),
                "resource_url": resource_url,
                "last_modified": package_metadata.get("last_modified") if package_metadata else None,
                "output_file": (
                    config["output_name"]
                    if dataset_name != "drc_subnational"
                    else "ebola_drc_subnational.csv"
                ),
                **info,
            }
            metadata["datasets"][dataset_name] = description
            log(
                f"{dataset_name}: {info['rows']:,} rows, "
                f"{info.get('start')} to {info.get('end')}"
            )

    with (data_dir / "ingestion_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log(f"Metadata written to {data_dir / 'ingestion_metadata.json'}")
    return metadata


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse command-line options and return a process exit code."""
    parser = argparse.ArgumentParser(description="Ingest Ebola outbreak data from HDX.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for raw and normalized data.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh downloads even when cached raw files exist.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached raw files without network requests.",
    )
    args = parser.parse_args()
    if args.refresh and args.offline:
        parser.error("--refresh and --offline cannot be combined")
    try:
        run(args.data_dir, args.refresh, args.offline)
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
