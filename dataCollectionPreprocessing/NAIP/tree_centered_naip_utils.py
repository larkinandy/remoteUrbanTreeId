"""Shared utilities for extracting tree-centered NAIP crops from source rasters.

The paired Sentinel-cell chip index records the natural-color and companion/CIR
SID files used for each reduced Sentinel cell. These helpers reuse those SID
paths and crop around a crown coordinate in the source CRS, avoiding recropping
from the smaller Sentinel-cell-centered .npy chips.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def spatial_reference_code(spatial_reference) -> int | None:
    for attr in ("factoryCode", "factory_code"):
        value = getattr(spatial_reference, attr, None)
        try:
            code = int(value)
        except (TypeError, ValueError):
            continue
        if code > 0:
            return code
    return None


def parse_path_rewrites(values: Iterable[str] | None) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in str(value):
            raise ValueError(f"Path rewrite must be OLD=NEW, got {value!r}")
        old, new = str(value).split("=", 1)
        rewrites.append((old, new))
    return rewrites


def resolve_existing_or_extract_sid(path: str | Path, rewrites: list[tuple[str, str]] | None = None) -> Path:
    candidate = Path(path)
    candidates = [candidate]
    for old, new in rewrites or []:
        text = str(candidate)
        if text.lower().startswith(old.lower()):
            candidates.append(Path(new + text[len(old) :]))
    for item in list(candidates):
        if item.exists():
            return item
        zip_path = item.with_suffix(".zip")
        if zip_path.exists():
            return extract_sid_from_zip(zip_path, item.name)
    raise FileNotFoundError(f"Could not find SID path or matching ZIP for {path}")


def extract_sid_from_zip(zip_path: Path, preferred_name: str | None = None) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        sid_entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".sid")]
        if not sid_entries:
            raise RuntimeError(f"No .sid file found in {zip_path}")
        if preferred_name:
            preferred = [entry for entry in sid_entries if Path(entry.filename).name.lower() == preferred_name.lower()]
            if preferred:
                sid_entries = preferred
        entry = sid_entries[0]
        destination = zip_path.parent / Path(entry.filename).name
        if destination.exists() and destination.stat().st_size == entry.file_size:
            return destination
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.unlink(missing_ok=True)
        with archive.open(entry) as source, partial.open("wb") as target:
            while True:
                chunk = source.read(16 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        if partial.stat().st_size != entry.file_size:
            raise RuntimeError(f"Extracted size mismatch for {destination}")
        partial.replace(destination)
        return destination


def load_sid_lookup(index_csv: Path, rewrites: list[tuple[str, str]] | None = None):
    index = pd.read_csv(index_csv, low_memory=False)
    required = {"row_index", "natural_sid", "companion_sid"}
    missing = required.difference(index.columns)
    if missing:
        raise RuntimeError(f"{index_csv} is missing required column(s): {sorted(missing)}")
    index = index.dropna(subset=["row_index", "natural_sid", "companion_sid"]).copy()
    index["row_index"] = index["row_index"].astype(np.int64)
    lookup = {}
    unique_pairs: list[tuple[Path, Path]] = []
    seen_pairs: set[tuple[Path, Path]] = set()
    resolved_cache: dict[str, Path] = {}
    for row in index.itertuples(index=False):
        natural_raw = str(getattr(row, "natural_sid"))
        companion_raw = str(getattr(row, "companion_sid"))
        if natural_raw not in resolved_cache:
            resolved_cache[natural_raw] = resolve_existing_or_extract_sid(natural_raw, rewrites)
        if companion_raw not in resolved_cache:
            resolved_cache[companion_raw] = resolve_existing_or_extract_sid(companion_raw, rewrites)
        pair = (resolved_cache[natural_raw], resolved_cache[companion_raw])
        lookup[int(row.row_index)] = pair
        if pair not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair)
    return lookup, unique_pairs


def load_5band_sid_lookup(index_csv: Path, rewrites: list[tuple[str, str]] | None = None):
    index = pd.read_csv(index_csv, low_memory=False)
    required = {"row_index", "source_sid", "red_band", "green_band", "blue_band", "nir_band"}
    missing = required.difference(index.columns)
    if missing:
        raise RuntimeError(f"{index_csv} is missing required column(s): {sorted(missing)}")
    index = index.dropna(subset=list(required)).copy()
    index["row_index"] = index["row_index"].astype(np.int64)
    lookup = {}
    unique_sources: list[tuple[Path, tuple[int, int, int, int]]] = []
    seen_sources: set[tuple[Path, tuple[int, int, int, int]]] = set()
    resolved_cache: dict[str, Path] = {}
    for row in index.itertuples(index=False):
        source_raw = str(getattr(row, "source_sid"))
        if source_raw not in resolved_cache:
            resolved_cache[source_raw] = resolve_existing_or_extract_sid(source_raw, rewrites)
        bands = (
            int(getattr(row, "red_band")),
            int(getattr(row, "green_band")),
            int(getattr(row, "blue_band")),
            int(getattr(row, "nir_band")),
        )
        source = (resolved_cache[source_raw], bands)
        lookup[int(row.row_index)] = source
        if source not in seen_sources:
            unique_sources.append(source)
            seen_sources.add(source)
    return lookup, unique_sources


def resize_bilinear(array: np.ndarray, height: int, width: int) -> np.ndarray:
    bands, source_h, source_w = array.shape
    if (source_h, source_w) == (height, width):
        return np.clip(array, 0, 255).astype(np.uint8)
    yy = np.linspace(0, source_h - 1, height)
    xx = np.linspace(0, source_w - 1, width)
    y0 = np.floor(yy).astype(int)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x0 = np.floor(xx).astype(int)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (yy - y0)[None, :, None]
    wx = (xx - x0)[None, None, :]
    top = array[:, y0, :][:, :, x0] * (1.0 - wx) + array[:, y0, :][:, :, x1] * wx
    bottom = array[:, y1, :][:, :, x0] * (1.0 - wx) + array[:, y1, :][:, :, x1] * wx
    return np.clip(top * (1.0 - wy) + bottom * wy, 0, 255).astype(np.uint8)


def chip_saturation_metrics(chip: np.ndarray) -> dict[str, float]:
    if chip.ndim != 3:
        raise ValueError("Expected chip with shape HxWxC")
    blackout = np.all(chip == 0, axis=2)
    whiteout = np.all(chip == 255, axis=2)
    saturated = blackout | whiteout
    return {
        "blackout_fraction": float(blackout.mean()),
        "whiteout_fraction": float(whiteout.mean()),
        "saturation_fraction": float(saturated.mean()),
        "valid_fraction": float((~saturated).mean()),
    }


class PairedNaipRasterCropper:
    def __init__(
        self,
        paired_index_csv: Path | None,
        source_epsg: int | str,
        rgb_bands: tuple[int, ...] = (1, 2, 3),
        nir_band: int = 1,
        path_rewrites: list[tuple[str, str]] | None = None,
        sid_pairs: list[tuple[Path, Path]] | None = None,
    ):
        import arcpy

        self.arcpy = arcpy
        self.source_epsg = int(str(source_epsg).replace("EPSG:", ""))
        self.source_sr = arcpy.SpatialReference(self.source_epsg)
        self.rgb_bands = tuple(int(value) for value in rgb_bands)
        self.nir_band = int(nir_band)
        if paired_index_csv is not None:
            self.sid_lookup, self.unique_sid_pairs = load_sid_lookup(paired_index_csv, path_rewrites)
        else:
            self.sid_lookup = {}
            self.unique_sid_pairs = [(Path(natural), Path(companion)) for natural, companion in sid_pairs or []]
            if not self.unique_sid_pairs:
                raise ValueError("No paired natural-color and companion NAIP sources were provided")
        self._raster_cache: dict[Path, object] = {}
        self._raster_meta: dict[Path, dict[str, object]] = {}

    def _raster(self, path: Path):
        if path not in self._raster_cache:
            raster = self.arcpy.Raster(str(path))
            if not raster.spatialReference or raster.spatialReference.name == "Unknown":
                raise RuntimeError(f"Missing CRS: {path}")
            self._raster_cache[path] = raster
            self._raster_meta[path] = {
                "epsg": spatial_reference_code(raster.spatialReference),
                "cell_x": abs(float(raster.meanCellWidth)),
                "cell_y": abs(float(raster.meanCellHeight)),
                "extent": raster.extent,
                "spatial_reference": raster.spatialReference,
                "band_count": int(getattr(raster, "bandCount", 1) or 1),
            }
        return self._raster_cache[path]

    def sid_pair_for_row(self, row_index: int) -> tuple[Path, Path]:
        if int(row_index) not in self.sid_lookup:
            raise KeyError(f"No paired NAIP SID lookup for row_index={row_index}")
        return self.sid_lookup[int(row_index)]

    def _project_point(self, x: float, y: float, raster_path: Path):
        meta = self._raster_meta[raster_path]
        if meta.get("epsg") == self.source_epsg:
            return self.arcpy.Point(float(x), float(y))
        source_point = self.arcpy.PointGeometry(self.arcpy.Point(float(x), float(y)), self.source_sr)
        projected_geom = source_point.projectAs(meta["spatial_reference"])
        if projected_geom is None:
            return None
        return projected_geom.firstPoint

    def _extract_resized_chip(self, dataset, raster_path: Path, projected_point, chip_metres: float, pixels: int, band_indexes: tuple[int, ...]) -> np.ndarray:
        meta = self._raster_meta[raster_path]
        cell_x = float(meta["cell_x"])
        cell_y = float(meta["cell_y"])
        source_cols = max(1, int(round(float(chip_metres) / cell_x)))
        source_rows = max(1, int(round(float(chip_metres) / cell_y)))
        lower_left = self.arcpy.Point(projected_point.X - chip_metres / 2.0, projected_point.Y - chip_metres / 2.0)
        chip = self.arcpy.RasterToNumPyArray(
            dataset,
            lower_left_corner=lower_left,
            ncols=source_cols,
            nrows=source_rows,
            nodata_to_value=0,
        )
        if chip.ndim == 2:
            chip = chip[None, :, :]
        max_band = max(band_indexes)
        if chip.shape[0] < max_band:
            raise RuntimeError(f"{dataset.catalogPath} has {chip.shape[0]} band(s), cannot read band {max_band}")
        chip = chip[[band - 1 for band in band_indexes], :, :]
        return resize_bilinear(chip.astype(np.float32, copy=False), pixels, pixels)

    def _block_spec(self, raster_path: Path, projected_points: list[object], chip_metres: float) -> dict[str, object]:
        meta = self._raster_meta[raster_path]
        cell_x = float(meta["cell_x"])
        cell_y = float(meta["cell_y"])
        source_cols = max(1, int(round(float(chip_metres) / cell_x)))
        source_rows = max(1, int(round(float(chip_metres) / cell_y)))
        half_width = source_cols * cell_x / 2.0
        half_height = source_rows * cell_y / 2.0
        xmin = min(point.X for point in projected_points) - half_width
        xmax = max(point.X for point in projected_points) + half_width
        ymin = min(point.Y for point in projected_points) - half_height
        ymax = max(point.Y for point in projected_points) + half_height
        extent = meta["extent"]
        xmin = max(float(extent.XMin), float(xmin))
        xmax = min(float(extent.XMax), float(xmax))
        ymin = max(float(extent.YMin), float(ymin))
        ymax = min(float(extent.YMax), float(ymax))
        ncols = max(source_cols, int(np.ceil((xmax - xmin) / cell_x)))
        nrows = max(source_rows, int(np.ceil((ymax - ymin) / cell_y)))
        xmax = xmin + ncols * cell_x
        ymax = ymin + nrows * cell_y
        return {
            "lower_left": self.arcpy.Point(xmin, ymin),
            "xmin": float(xmin),
            "ymax": float(ymax),
            "ncols": int(ncols),
            "nrows": int(nrows),
            "source_cols": int(source_cols),
            "source_rows": int(source_rows),
            "cell_x": float(cell_x),
            "cell_y": float(cell_y),
            "band_count": int(meta.get("band_count", 1) or 1),
        }

    def _estimated_block_bytes(self, spec: dict[str, object]) -> int:
        return int(spec["ncols"]) * int(spec["nrows"]) * int(spec["band_count"])

    def _read_block(self, dataset, spec: dict[str, object], band_indexes: tuple[int, ...]) -> np.ndarray:
        block = self.arcpy.RasterToNumPyArray(
            dataset,
            lower_left_corner=spec["lower_left"],
            ncols=int(spec["ncols"]),
            nrows=int(spec["nrows"]),
            nodata_to_value=0,
        )
        if block.ndim == 2:
            block = block[None, :, :]
        max_band = max(band_indexes)
        if block.shape[0] < max_band:
            raise RuntimeError(f"{dataset.catalogPath} has {block.shape[0]} band(s), cannot read band {max_band}")
        return block[[band - 1 for band in band_indexes], :, :]

    def _point_inside_raster(self, raster_path: Path, projected_point) -> bool:
        extent = self._raster_meta[raster_path]["extent"]
        return bool(
            extent.XMin <= projected_point.X <= extent.XMax
            and extent.YMin <= projected_point.Y <= extent.YMax
        )

    def _crop_from_block(self, block: np.ndarray, spec: dict[str, object], projected_point, pixels: int) -> np.ndarray:
        cell_x = float(spec["cell_x"])
        cell_y = float(spec["cell_y"])
        source_cols = int(spec["source_cols"])
        source_rows = int(spec["source_rows"])
        center_col = int(round((float(projected_point.X) - float(spec["xmin"])) / cell_x))
        center_row = int(round((float(spec["ymax"]) - float(projected_point.Y)) / cell_y))
        col0 = center_col - source_cols // 2
        row0 = center_row - source_rows // 2
        col1 = col0 + source_cols
        row1 = row0 + source_rows
        src_col0 = max(0, col0)
        src_row0 = max(0, row0)
        src_col1 = min(block.shape[2], col1)
        src_row1 = min(block.shape[1], row1)
        chip = np.zeros((block.shape[0], source_rows, source_cols), dtype=block.dtype)
        dst_col0 = src_col0 - col0
        dst_row0 = src_row0 - row0
        dst_col1 = dst_col0 + max(0, src_col1 - src_col0)
        dst_row1 = dst_row0 + max(0, src_row1 - src_row0)
        if dst_col1 > dst_col0 and dst_row1 > dst_row0:
            chip[:, dst_row0:dst_row1, dst_col0:dst_col1] = block[:, src_row0:src_row1, src_col0:src_col1]
        return resize_bilinear(chip.astype(np.float32, copy=False), pixels, pixels)

    def crop_rgbnir_batch(
        self,
        rows: list[tuple[int, float, float]],
        chip_metres: float,
        pixels: int,
        max_block_gb: float = 48.0,
    ) -> list[tuple[np.ndarray, str, str]]:
        if not rows:
            return []
        natural_sid, companion_sid = self.sid_pair_for_row(rows[0][0])
        for row_index, _x, _y in rows:
            if self.sid_pair_for_row(row_index) != (natural_sid, companion_sid):
                raise ValueError("crop_rgbnir_batch requires rows from a single SID pair")
        explicit_results = self.crop_rgbnir_batch_from_pair(natural_sid, companion_sid, rows, chip_metres, pixels, max_block_gb)
        return [(crop, natural_text, companion_text) for _row_index, crop, natural_text, companion_text in explicit_results]

    def crop_rgbnir_batch_from_pair(
        self,
        natural_sid: Path,
        companion_sid: Path,
        rows: list[tuple[int, float, float]],
        chip_metres: float,
        pixels: int,
        max_block_gb: float = 48.0,
    ) -> list[tuple[int, np.ndarray, str, str]]:
        if not rows:
            return []
        natural = self._raster(natural_sid)
        companion = self._raster(companion_sid)
        projected_nat = []
        projected_cir = []
        kept_rows: list[tuple[int, float, float]] = []
        for row_index, center_x, center_y in rows:
            nat_point = self._project_point(float(center_x), float(center_y), natural_sid)
            cir_point = self._project_point(float(center_x), float(center_y), companion_sid)
            if nat_point is None or cir_point is None:
                raise RuntimeError(f"Could not project point for row_index={row_index}")
            if not self._point_inside_raster(natural_sid, nat_point) or not self._point_inside_raster(companion_sid, cir_point):
                continue
            projected_nat.append(nat_point)
            projected_cir.append(cir_point)
            kept_rows.append((row_index, center_x, center_y))
        if not projected_nat:
            return []
        nat_spec = self._block_spec(natural_sid, projected_nat, chip_metres)
        cir_spec = self._block_spec(companion_sid, projected_cir, chip_metres)
        estimated_gb = (self._estimated_block_bytes(nat_spec) + self._estimated_block_bytes(cir_spec)) / (1024.0**3)
        if estimated_gb > float(max_block_gb):
            raise MemoryError(f"SID-pair block estimate {estimated_gb:.1f} GB exceeds --max-block-gb={max_block_gb}")
        rgb_block = self._read_block(natural, nat_spec, self.rgb_bands)
        nir_block = self._read_block(companion, cir_spec, (self.nir_band,))
        results: list[tuple[int, np.ndarray, str, str]] = []
        for position, (row_index, _center_x, _center_y) in enumerate(kept_rows):
            rgb = self._crop_from_block(rgb_block, nat_spec, projected_nat[position], pixels)
            nir = self._crop_from_block(nir_block, cir_spec, projected_cir[position], pixels)
            chip = np.empty((pixels, pixels, 4), dtype=np.uint8)
            chip[:, :, 0:3] = np.moveaxis(rgb[:3], 0, -1)
            chip[:, :, 3] = nir[0]
            results.append((row_index, chip, str(natural_sid), str(companion_sid)))
        return results

    def crop_rgbnir(self, row_index: int, center_x: float, center_y: float, chip_metres: float, pixels: int) -> tuple[np.ndarray, str, str]:
        natural_sid, companion_sid = self.sid_pair_for_row(row_index)
        return self.crop_rgbnir_from_pair(natural_sid, companion_sid, row_index, center_x, center_y, chip_metres, pixels)

    def crop_rgbnir_from_pair(
        self,
        natural_sid: Path,
        companion_sid: Path,
        row_index: int,
        center_x: float,
        center_y: float,
        chip_metres: float,
        pixels: int,
    ) -> tuple[np.ndarray, str, str]:
        natural = self._raster(natural_sid)
        companion = self._raster(companion_sid)
        projected_nat = self._project_point(float(center_x), float(center_y), natural_sid)
        projected_cir = self._project_point(float(center_x), float(center_y), companion_sid)
        if projected_nat is None or projected_cir is None:
            raise RuntimeError(f"Could not project point for row_index={row_index}")
        nat_extent = self._raster_meta[natural_sid]["extent"]
        cir_extent = self._raster_meta[companion_sid]["extent"]
        if not (
            nat_extent.XMin <= projected_nat.X <= nat_extent.XMax
            and nat_extent.YMin <= projected_nat.Y <= nat_extent.YMax
            and cir_extent.XMin <= projected_cir.X <= cir_extent.XMax
            and cir_extent.YMin <= projected_cir.Y <= cir_extent.YMax
        ):
            raise RuntimeError(f"Projected point falls outside paired raster extent for row_index={row_index}")
        rgb = self._extract_resized_chip(natural, natural_sid, projected_nat, chip_metres, pixels, self.rgb_bands)
        nir = self._extract_resized_chip(companion, companion_sid, projected_cir, chip_metres, pixels, (self.nir_band,))
        chip = np.empty((pixels, pixels, 4), dtype=np.uint8)
        chip[:, :, 0:3] = np.moveaxis(rgb[:3], 0, -1)
        chip[:, :, 3] = nir[0]
        return chip, str(natural_sid), str(companion_sid)

    def crop_rgbnir_best_overlap(
        self,
        row_index: int,
        center_x: float,
        center_y: float,
        chip_metres: float,
        pixels: int,
    ) -> tuple[np.ndarray, str, str, dict[str, float]]:
        best: tuple[float, float, np.ndarray, str, str, dict[str, float]] | None = None
        errors = []
        for natural_sid, companion_sid in self.unique_sid_pairs:
            try:
                chip, natural_text, companion_text = self.crop_rgbnir_from_pair(
                    natural_sid,
                    companion_sid,
                    row_index,
                    center_x,
                    center_y,
                    chip_metres,
                    pixels,
                )
            except Exception as error:
                if len(errors) < 3:
                    errors.append(str(error))
                continue
            metrics = chip_saturation_metrics(chip)
            score = float(metrics["saturation_fraction"])
            valid = float(metrics["valid_fraction"])
            candidate = (score, -valid, chip, natural_text, companion_text, metrics)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            detail = "; ".join(errors) if errors else "no candidate SID pairs"
            raise RuntimeError(f"No overlapping usable SID pair for row_index={row_index}: {detail}")
        _score, _neg_valid, chip, natural_text, companion_text, metrics = best
        return chip, natural_text, companion_text, metrics

    def close(self) -> None:
        self._raster_cache.clear()


class FiveBandNaipRasterCropper(PairedNaipRasterCropper):
    def __init__(
        self,
        index_csv: Path | None,
        source_epsg: int | str,
        path_rewrites: list[tuple[str, str]] | None = None,
        sid_sources: list[Path] | None = None,
        bands: tuple[int, int, int, int] = (1, 2, 3, 4),
    ):
        import arcpy

        self.arcpy = arcpy
        self.source_epsg = int(str(source_epsg).replace("EPSG:", ""))
        self.source_sr = arcpy.SpatialReference(self.source_epsg)
        self.rgb_bands = (1, 2, 3)
        self.nir_band = 4
        if index_csv is not None:
            self.sid_lookup, self.unique_sid_sources = load_5band_sid_lookup(index_csv, path_rewrites)
        else:
            self.sid_lookup = {}
            self.unique_sid_sources = [(Path(path), tuple(int(value) for value in bands)) for path in sid_sources or []]
            if not self.unique_sid_sources:
                raise ValueError("No 5-band SID sources were provided")
        self._raster_cache: dict[Path, object] = {}
        self._raster_meta: dict[Path, dict[str, object]] = {}

    def sid_source_for_row(self, row_index: int) -> tuple[Path, tuple[int, int, int, int]]:
        if int(row_index) not in self.sid_lookup:
            raise KeyError(f"No 5-band NAIP SID lookup for row_index={row_index}")
        return self.sid_lookup[int(row_index)]

    def crop_rgbnir_batch(
        self,
        rows: list[tuple[int, float, float]],
        chip_metres: float,
        pixels: int,
        max_block_gb: float = 48.0,
    ) -> list[tuple[np.ndarray, str, tuple[int, int, int, int]]]:
        if not rows:
            return []
        source_sid, bands = self.sid_source_for_row(rows[0][0])
        for row_index, _x, _y in rows:
            if self.sid_source_for_row(row_index) != (source_sid, bands):
                raise ValueError("crop_rgbnir_batch requires rows from a single 5-band SID source")
        explicit_results = self.crop_rgbnir_batch_from_source(source_sid, bands, rows, chip_metres, pixels, max_block_gb)
        return [(crop, source_text, source_bands) for _row_index, crop, source_text, source_bands in explicit_results]

    def crop_rgbnir_batch_from_source(
        self,
        source_sid: Path,
        bands: tuple[int, int, int, int],
        rows: list[tuple[int, float, float]],
        chip_metres: float,
        pixels: int,
        max_block_gb: float = 48.0,
    ) -> list[tuple[int, np.ndarray, str, tuple[int, int, int, int]]]:
        if not rows:
            return []
        source = self._raster(source_sid)
        projected = []
        kept_rows: list[tuple[int, float, float]] = []
        for row_index, center_x, center_y in rows:
            point = self._project_point(float(center_x), float(center_y), source_sid)
            if point is None:
                raise RuntimeError(f"Could not project point for row_index={row_index}")
            if not self._point_inside_raster(source_sid, point):
                continue
            projected.append(point)
            kept_rows.append((row_index, center_x, center_y))
        if not projected:
            return []
        spec = self._block_spec(source_sid, projected, chip_metres)
        estimated_gb = self._estimated_block_bytes(spec) / (1024.0**3)
        if estimated_gb > float(max_block_gb):
            raise MemoryError(f"5-band SID block estimate {estimated_gb:.1f} GB exceeds --max-block-gb={max_block_gb}")
        source_block = self._read_block(source, spec, tuple(int(value) for value in bands))
        results: list[tuple[int, np.ndarray, str, tuple[int, int, int, int]]] = []
        for position, (row_index, _center_x, _center_y) in enumerate(kept_rows):
            chip_bands = self._crop_from_block(source_block, spec, projected[position], pixels)
            chip = np.empty((pixels, pixels, 4), dtype=np.uint8)
            chip[:, :, :] = np.moveaxis(chip_bands[:4], 0, -1)
            results.append((row_index, chip, str(source_sid), tuple(int(value) for value in bands)))
        return results

    def crop_rgbnir(self, row_index: int, center_x: float, center_y: float, chip_metres: float, pixels: int) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
        source_sid, bands = self.sid_source_for_row(row_index)
        return self.crop_rgbnir_from_source(source_sid, bands, row_index, center_x, center_y, chip_metres, pixels)

    def crop_rgbnir_from_source(
        self,
        source_sid: Path,
        bands: tuple[int, int, int, int],
        row_index: int,
        center_x: float,
        center_y: float,
        chip_metres: float,
        pixels: int,
    ) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
        source = self._raster(source_sid)
        projected = self._project_point(float(center_x), float(center_y), source_sid)
        if projected is None:
            raise RuntimeError(f"Could not project point for row_index={row_index}")
        if not self._point_inside_raster(source_sid, projected):
            raise RuntimeError(f"Projected point falls outside 5-band raster extent for row_index={row_index}")
        chip_bands = self._extract_resized_chip(source, source_sid, projected, chip_metres, pixels, tuple(int(value) for value in bands))
        chip = np.empty((pixels, pixels, 4), dtype=np.uint8)
        chip[:, :, :] = np.moveaxis(chip_bands[:4], 0, -1)
        return chip, str(source_sid), tuple(int(value) for value in bands)

    def crop_rgbnir_best_overlap(
        self,
        row_index: int,
        center_x: float,
        center_y: float,
        chip_metres: float,
        pixels: int,
    ) -> tuple[np.ndarray, str, tuple[int, int, int, int], dict[str, float]]:
        best: tuple[float, float, np.ndarray, str, tuple[int, int, int, int], dict[str, float]] | None = None
        errors = []
        for source_sid, bands in self.unique_sid_sources:
            try:
                chip, source_text, source_bands = self.crop_rgbnir_from_source(
                    source_sid,
                    bands,
                    row_index,
                    center_x,
                    center_y,
                    chip_metres,
                    pixels,
                )
            except Exception as error:
                if len(errors) < 3:
                    errors.append(str(error))
                continue
            metrics = chip_saturation_metrics(chip)
            score = float(metrics["saturation_fraction"])
            valid = float(metrics["valid_fraction"])
            candidate = (score, -valid, chip, source_text, source_bands, metrics)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            detail = "; ".join(errors) if errors else "no candidate SID sources"
            raise RuntimeError(f"No overlapping usable 5-band SID source for row_index={row_index}: {detail}")
        _score, _neg_valid, chip, source_text, source_bands, metrics = best
        return chip, source_text, source_bands, metrics
