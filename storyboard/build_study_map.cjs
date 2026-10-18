// Build an offline SVG using d3-geo and published Census-based US Atlas geometry.
// Usage: node storyboard/build_study_map.cjs <input directory> <city JSON>
// Input directory: d3.cjs (7.9.0), topojson.cjs (3.1.0), us-states-10m.json.
const fs = require('fs');
const path = require('path');
const dir = path.resolve(process.argv[2]);
const d3 = require(path.join(dir, 'd3.cjs'));
const topojson = require(path.join(dir, 'topojson.cjs'));
const cities = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const atlas = JSON.parse(fs.readFileSync(path.join(dir, 'us-states-10m.json'), 'utf8'));
const states = topojson.feature(atlas, atlas.objects.states);
states.features = states.features.filter(s => !['02', '15', '72'].includes(s.id));
const projection = d3.geoAlbersUsa().fitExtent([[12, 12], [628, 378]], states);
const geoPath = d3.geoPath(projection).digits(1);
const paths = states.features.map(s => `<path d="${geoPath(s)}"/>`).join('');
const markers = cities.map(c => {
  const [x, y] = projection([c.longitude, c.latitude]);
  return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" data-study-city="${c.token}"><title>${c.city}, ${c.state}: ${c.records.toLocaleString('en-US')} trees</title></circle>`;
}).join('');
const svg = `<svg class="sb-city-map" viewBox="0 0 640 390" role="img" aria-label="Locations of the 36 study cities in the contiguous United States"><g class="sb-map-states">${paths}</g><g class="sb-map-markers">${markers}</g></svg>`;
fs.writeFileSync(path.join(__dirname, 'assets/study-cities.svg'), svg);
