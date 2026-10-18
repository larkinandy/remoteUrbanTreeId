# Background photograph

## Current photograph

**A shady corner of Bramley Park** by Stephen Craven, photographed 10 August
2026 in Leeds, England.

Source: https://www.geograph.org.uk/photo/8381778

Creative Commons Attribution-ShareAlike 2.0:
https://creativecommons.org/licenses/by-sa/2.0/

`tree-shade-bramley.jpg` preserves the downloaded image.
`tree-shade-bramley-web.jpg` is JPEG-compressed for display without cropping or
content changes. Both remain under CC BY-SA 2.0. The web image is embedded in
the storyboard with a visible credit and license link.

## Previous photograph

**Tree-8748.jpg** by David R. Tribble (Wikimedia username: Loadmaster).
Photographed 17 April 2010 at Bob Woodruff Park, Plano, Texas.

Source: https://commons.wikimedia.org/wiki/File:Tree-8748.jpg

Used under Creative Commons Attribution-ShareAlike 3.0 Unported:
https://creativecommons.org/licenses/by-sa/3.0/

`tree-shade-plano.jpg` preserves the downloaded source image.
`tree-shade-plano-web.jpg` is resized and JPEG-compressed for display, with no
cropping or content changes. Both photographs remain under CC BY-SA 3.0.
The optimized photograph is embedded in `story.html` with a visible attribution
and license link. It illustrates urban nature; it is not a study observation.
# Study-city map

`anaheim-tree-63599-prism-daily.png` plots the `ppt` (mm), `tmean`
(degrees Celsius), and `vpdmax` (hPa) inputs for Anaheim tree_id 63599,
Auranticarpa rhombifolia. It uses the full daily sidecar at
`H:/tree_centered_prism_daily_clean/prism_daily_values.npz`, linked through
the Anaheim index by tree_id. The selected PRISM pixel is grid row 386,
column 169 (global array source row 1215). All 1,095 daily observations from
2021-01-01 through 2023-12-31 are valid and plotted. Precipitation is displayed
on a 0–10 mm axis, with a plus sign for each value above 10 mm; source values
remain unchanged. Both figures use the January 2021 to January 2024 axis range.
Exact plotted values are in `anaheim-tree-63599-prism-daily.csv`. The source script
is `storyboard/build_weather_example.py`; PRISM units follow the frozen study
configuration in `dataCollectionPreprocessing/PRISM/provenance/`.

`portland-south-park-blocks.jpg`: “South Park Blocks” by Visitor7, photographed
20 April 2013 in Portland, Oregon. Source:
https://commons.wikimedia.org/wiki/File:South_Park_Blocks.jpg
Licensed CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
Resized and JPEG-compressed, without cropping or content alteration. The
optimized photograph remains under CC BY-SA 3.0 and is embedded in the
Conclusions section with photographer credit and source/license links.
It illustrates an urban streetscape and is not a study observation.

`tree-crown-matching-s4-bottom.jpg` is the complete lower example cropped from
repository `images/treeCrown.png`, corresponding to Figure S4's crown matching
example for Koelreuteria paniculata. It retains all three matching stages,
the taxon label, match distance, and scale bars. A caption restores the marker
legend from the full figure, which is outside the excerpt.

`denver-red-oak-false-color.jpg` is the Denver northern red oak NIR/red/green
panel cropped from repository `images/NAIP.png` (Supplementary Figure S6).
It is retained as a source asset. `denver-red-oak-ndvi.jpg`, cropped from the
same row's NDVI panel in Figure S6, now accompanies the RGB view of the same
tree in the aerial imagery panel.

`denver-red-oak-raw-lidar-returns.png` shows the raw point returns within the
same 38 x 38 m crown-centered footprint as the Denver CHM example in
`images/CHM.png`: Quercus rubra, tree_id 52199, crown EPSG:32613 coordinates
507005.3, 4397053.5. The 9,118 returns were extracted from
`E:/tree_datasets/LiDAR/DEN/USGS_LPC_CO_DRCOG_2020_B20_w0505n4396.laz` and
`USGS_LPC_CO_DRCOG_2020_B20_w0507n4396.laz`. Dots are colored by raw elevation;
the conceptual panel omits axes and a legend. It replaces the first aerial
image only when the LiDAR dataset tab is selected. No interpolation, height
normalization, or vegetation masking was applied. Point provenance is saved in
`outputs/manuscript_supplement/raw_lidar_denver_tree_52199.json`.

`study-cities.json` joins the 36 city tokens to user-supplied filtered study
counts preserved in `study-city-filtered-records.csv`, and representative place
coordinates from the [2024 U.S. Census Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2024.html).
Arlington is Texas; Aurora is Colorado; Portland is Oregon; Rochester is New York.
The city counts include both core and rare taxa from the final stage-two split
table, `outputs/reviewer_revision/stage2_partitions_k11_groupnorm_v1/records_by_split_tier_city.csv`.
Summing across training, validation, and test partitions gives 1,794,468
core-taxon records plus 57,500 rare-taxon records: 1,851,968 records in total.
The CSV retains separate `core_records` and `rare_records` columns, and
`filtered_records` is their sum. Run `storyboard/update_study_city_counts.py`
to synchronize the CSV, city JSON, SVG titles, and embedded storyboard data.
The supplied table labels the Iowa inventory as Newton (7,559 filtered records),
while the repository uses the internal token `desmoines` for its 16,875 raw records.
The map uses Newton's name and Census coordinates, retaining that internal token.

`study-cities.svg` uses Census-derived state boundaries from
[US Atlas 3](https://github.com/topojson/us-atlas) (ISC license), projected with
D3 7.9.0 using `geoAlbersUsa`. It is embedded in the storyboard for offline use.
`storyboard/build_study_map.cjs` documents the map-generation command and inputs.
Coordinates mark reference city locations, not individual trees or inventory extents.

