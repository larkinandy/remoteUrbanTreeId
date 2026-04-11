# script to download environmental estimates from the Google Earth Engine
# Author: Andrew Larkin
# Date created: April 11, 2026
# ChatGPT was used for code assist

# import libraries
import ee
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv


# import custom classes and environments
GIT_PATH = "C:/Users/larki/Documents/GitHub/remoteUrbanTreeId/dataCollection/"
sys.path.append(GIT_PATH)

load_dotenv(dotenv_path=GIT_PATH + ".env")

ee.Authenticate()
ee.Initialize(os.getenv("GEE_PROJECT")

INTERVAL_DAYS = 7

BANDS = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
    'B8', 'B8A', 'B11', 'B12'
]

CLOUD_FILTER = 70        # scene-level filter
CLD_PRB_THRESH = 40      # s2cloudless threshold, 0-100

# SCL classes to mask out:
# 0 = No Data
# 1 = Saturated/Defective
# 3 = Cloud Shadow
# 8 = Cloud medium probability
# 9 = Cloud high probability
# 10 = Thin cirrus
# 11 = Snow/Ice
BAD_SCL_CLASSES = [0, 1, 3, 8, 9, 10, 11]

points = ee.FeatureCollection(ee.FeatureCollection(os.getenv("AA_TREE_ASSET")))

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2025, 1, 1)

# ----------------------------------------
# LOAD + JOIN COLLECTIONS FOR A DATE WINDOW
# ----------------------------------------
def get_joined_collection(start_str, end_str):
    s2_sr = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_str, end_str)
        .filterBounds(points)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
    )

    s2_clouds = (
        ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterDate(start_str, end_str)
        .filterBounds(points)
    )

    joined = ee.ImageCollection(
        ee.Join.saveFirst('s2cloudless').apply(
            primary=s2_sr,
            secondary=s2_clouds,
            condition=ee.Filter.equals(
                leftField='system:index',
                rightField='system:index'
            )
        )
    )

    return joined


# ----------------------------------------
# FAST MASK FUNCTION
# ----------------------------------------
def mask_s2_fast(img):
    # Cloud probability image from joined collection
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    cloud_mask = cld_prb.lt(CLD_PRB_THRESH)

    # SCL-based mask
    scl = img.select('SCL')

    # Start with "all good", then remove bad classes
    scl_good = ee.Image(1)
    for cls in BAD_SCL_CLASSES:
        scl_good = scl_good.And(scl.neq(cls))

    # Combine masks
    mask = cloud_mask.And(scl_good)

    # Keep only spectral bands for export
    out = (
        img.updateMask(mask)
        .select(BANDS)
        .resample('bilinear')
        .copyProperties(img, img.propertyNames())
    )

    return (out.set('date', img.date().format('YYYY-MM-dd')))
   

# ----------------------------------------
# SAMPLE ONE IMAGE
# ----------------------------------------
def sample_one_image(img):
    samples = img.sampleRegions(
        collection=points,
        scale=10,
        geometries=False
    )

    def add_meta(f):
        return (
            f.set('date', img.get('date'))
        )

    return samples.map(add_meta)



# ----------------------------------------
# EXPORT LOOP
# ----------------------------------------
tasks = []

current = START_DATE
while current < END_DATE:
    window_end = min(current + timedelta(days=INTERVAL_DAYS), END_DATE)

    start_str = current.strftime('%Y-%m-%d')
    end_str = window_end.strftime('%Y-%m-%d')

    print(f'Preparing export: {start_str} to {end_str}')

    joined = get_joined_collection(start_str, end_str)
    masked = joined.map(mask_s2_fast)
    table = masked.map(sample_one_image).flatten()

    description = f'S2_fast_points_{start_str}'
    filename = f's2_fast_points_{start_str}'

    task = ee.batch.Export.table.toDrive(
        collection=table,
        description=description,
        folder=os.getenv("GEE_TREE_FOLDER"),
        fileNamePrefix=filename,
        fileFormat='CSV',
        selectors=['uniqueID', 'date'] + BANDS
        # Add your point ID here if it exists, e.g.:
        # selectors=['point_id', 'image_id', 'date', 'datetime'] + BANDS
    )

    task.start()
    tasks.append(task)

    print(f'Started task: {description}')
    current = window_end

print(f'Started {len(tasks)} export tasks.')
for t in tasks:
    print(t.status())