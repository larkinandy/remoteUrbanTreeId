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

`study-cities.json` joins the 36 city tokens to user-supplied filtered study
counts preserved in `study-city-filtered-records.csv`, and representative place
coordinates from the [2024 U.S. Census Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2024.html).
Arlington is Texas; Aurora is Colorado; Portland is Oregon; Rochester is New York.
The supplied filtered city counts sum to 1,794,468 records, which is 57,500 fewer
than the repository's analytical total of 1,851,968 records. This discrepancy
remains unresolved; the storyboard's overall total has not been changed.
The supplied table labels the Iowa inventory as Newton (7,559 filtered records),
while the repository uses the internal token `desmoines` for its 16,875 raw records.
The map uses Newton's name and Census coordinates, retaining that internal token.

`study-cities.svg` uses Census-derived state boundaries from
[US Atlas 3](https://github.com/topojson/us-atlas) (ISC license), projected with
D3 7.9.0 using `geoAlbersUsa`. It is embedded in the storyboard for offline use.
`storyboard/build_study_map.cjs` documents the map-generation command and inputs.
Coordinates mark reference city locations, not individual trees or inventory extents.

