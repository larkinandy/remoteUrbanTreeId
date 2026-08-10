from __future__ import annotations

import sys

import arcpy


def main() -> int:
    path = sys.argv[1]
    raster = arcpy.Raster(path)
    desc = arcpy.Describe(path)
    print(f"path={path}")
    print(f"bandCount={raster.bandCount}")
    print(f"pixelType={raster.pixelType}")
    print(f"meanCellWidth={raster.meanCellWidth}")
    print(f"meanCellHeight={raster.meanCellHeight}")
    print(f"format={getattr(desc, 'format', '')}")
    print(f"spatialReference={desc.spatialReference.name}")
    for i in range(1, int(raster.bandCount) + 1):
        band = arcpy.Raster(f"{path}/Band_{i}")
        print(
            f"Band_{i}: pixelType={band.pixelType}, "
            f"minimum={band.minimum}, maximum={band.maximum}, "
            f"mean={band.mean}, std={band.standardDeviation}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
