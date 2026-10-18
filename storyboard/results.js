// Offline results interactions. Study data are embedded as JSON in story.html.
(() => {
  const root = document.getElementById('urban-forest-storyboard');
  const data = JSON.parse(root.querySelector('#sb-results-data').textContent);
  const ns = 'http://www.w3.org/2000/svg';
  const fmt = value => value.toLocaleString('en-US');
  const percent = value => (value * 100).toFixed(1) + '%';
  const matrix = root.querySelector('.sb-confusion-grid');
  const counts = data.confusion.counts;
  const totals = counts.map(row => row.reduce((a, b) => a + b, 0));
  const corner = document.createElement('span');
  corner.className = 'sb-matrix-corner';
  corner.textContent = 'Labeled ↓';
  matrix.append(corner);
  for (let j = 0; j < 6; j++) {
    const label = document.createElement('span');
    label.className = 'sb-matrix-heading';
    label.textContent = 'Group ' + (j + 1);
    matrix.append(label);
  }
  counts.forEach((row, i) => {
    const label = document.createElement('span');
    label.className = 'sb-matrix-heading';
    label.textContent = 'Group ' + (i + 1);
    matrix.append(label);
    row.forEach((n, j) => {
      const cell = document.createElement('div');
      const share = n / totals[i];
      cell.className = 'sb-matrix-cell';
      cell.style.setProperty('--sb-cell-share', (share * 100).toFixed(3) + '%');
      cell.dataset.strong = String(share >= 0.5);
      cell.innerHTML = `<strong>${percent(share)}</strong><span>(${fmt(n)})</span>`;
      cell.setAttribute('aria-label', `Labeled Group ${i + 1}, predicted Group ${j + 1}: ${fmt(n)} records, ${percent(share)} of labeled group`);
      matrix.append(cell);
    });
  });

  const regimes = [
    {name: 'Default multimodal', short: 'Default', color: 'var(--sb-geo-default)', shape: 'circle'},
    {name: 'Leave one city out', short: 'Leave-city-out', color: 'var(--sb-geo-loco)', shape: 'square'},
    {name: 'One-city trained', short: 'One-city', color: 'var(--sb-geo-one)', shape: 'diamond'}
  ];
  const metrics = [['accuracy', 'Accuracy'], ['macro_f1', 'Macro-F1'], ['weighted_f1', 'Weighted-F1']];
  const plotHost = root.querySelector('.sb-geographic-plots');
  const geoSection = root.querySelector('.sb-geographic-analysis');
  const tooltip = document.createElement('div');
  tooltip.className = 'sb-geo-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.hidden = true;
  geoSection.append(tooltip);
  const metricValue = (value, metric) => metric === 'accuracy' ? percent(value) : value.toFixed(3);
  function showTooltip(row, metric, event, mark) {
    const city = document.createElement('strong');
    city.textContent = cityName(row.target_city);
    const regime = document.createElement('span');
    regime.textContent = row.model;
    const value = document.createElement('span');
    value.textContent = `${metrics.find(m => m[0] === metric)[1]}: ${metricValue(row[metric], metric)}`;
    tooltip.replaceChildren(city, regime, value);
    tooltip.hidden = false;
    const bounds = geoSection.getBoundingClientRect();
    const point = mark.getBoundingClientRect();
    const x = event?.clientX ?? point.right;
    const y = event?.clientY ?? point.top;
    tooltip.style.left = Math.max(0, Math.min(x - bounds.left + 12, geoSection.clientWidth - tooltip.offsetWidth)) + 'px';
    const below = y - bounds.top + 14;
    tooltip.style.top = (below + tooltip.offsetHeight > geoSection.clientHeight ? Math.max(0, y - bounds.top - tooltip.offsetHeight - 10) : below) + 'px';
  }
  function hideTooltip() { tooltip.hidden = true; }
  let selectedCity = null;
  let cityMarks = [];
  let pinnedPoint = null;
  const metricBoxes = [];
  function cityName(token) { return data.cities[token] || token; }
  function highlightCity(city) {
    cityMarks.forEach(mark => {
      mark.setAttribute('opacity', !city || mark.dataset.city === city ? '1' : '.18');
      mark.setAttribute('stroke-width', mark.dataset.city === city ? '2.5' : '1');
    });
  }
  function resetHighlight() { highlightCity(selectedCity); }
  function el(tag, attrs = {}, text = '') {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  }
  function quantile(sorted, p) {
    const position = (sorted.length - 1) * p;
    const low = Math.floor(position), delta = position - low;
    return sorted[low] + delta * ((sorted[low + 1] ?? sorted[low]) - sorted[low]);
  }
  function draw() {
    hideTooltip();
    cityMarks = [];
    metricBoxes.forEach(({box, metric, label}) => {
      const width = box.clientWidth;
      if (width < 100) return;
      const height = 365, pad = {l: 48, r: 14, t: 14, b: 56};
      const inner = width - pad.l - pad.r;
      const y = value => pad.t + (1 - value) * (height - pad.t - pad.b);
      box.replaceChildren();
      const svg = el('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${label}: city-level distributions for the default, leave-one-city-out, and one-city models`});
      box.append(svg);
      for (let tick = 0; tick <= 1.001; tick += .25) {
        svg.append(el('line', {x1: pad.l, y1: y(tick), x2: width - pad.r, y2: y(tick), stroke: 'var(--sb-line)', 'stroke-width': 1}));
        svg.append(el('text', {x: pad.l - 8, y: y(tick) + 4, 'text-anchor': 'end', fill: 'var(--sb-muted)', 'font-size': 11}, tick.toFixed(2)));
      }
      svg.append(el('text', {class: 'axis-title', 'data-axis': 'y', transform: `translate(13 ${y(.5)}) rotate(-90)`, 'text-anchor': 'middle', fill: 'var(--sb-muted)', 'font-size': 11}, 'Test-set score'));
      svg.append(el('text', {class: 'axis-title', 'data-axis': 'x', x: pad.l + inner / 2, y: height - 5, 'text-anchor': 'middle', fill: 'var(--sb-muted)', 'font-size': 11}, 'Training regime'));
      regimes.forEach((regime, index) => {
        const observations = data.geographic.filter(row => row.model === regime.name);
        const values = observations.map(row => row[metric]).sort((a, b) => a - b);
        const q1 = quantile(values, .25), median = quantile(values, .5), q3 = quantile(values, .75), iqr = q3 - q1;
        const lower = values.find(value => value >= q1 - 1.5 * iqr);
        const upper = [...values].reverse().find(value => value <= q3 + 1.5 * iqr);
        const x = pad.l + inner * (index + .5) / 3;
        const bw = Math.min(35, inner / 7);
        const boxGroup = el('g', {'aria-hidden': 'true'});
        boxGroup.append(el('line', {x1: x, x2: x, y1: y(lower), y2: y(upper), stroke: 'var(--sb-muted)'}));
        [lower, upper].forEach(value => boxGroup.append(el('line', {x1: x - bw / 3, x2: x + bw / 3, y1: y(value), y2: y(value), stroke: 'var(--sb-muted)'})));
        boxGroup.append(el('rect', {x: x - bw / 2, y: y(q3), width: bw, height: Math.max(1, y(q1) - y(q3)), fill: regime.color, 'fill-opacity': .25, stroke: regime.color}));
        boxGroup.append(el('line', {x1: x - bw / 2, x2: x + bw / 2, y1: y(median), y2: y(median), stroke: 'var(--sb-ink)', 'stroke-width': 2}));
        svg.append(boxGroup);
        svg.append(el('text', {x, y: height - 31, 'text-anchor': 'middle', fill: 'var(--sb-muted)', 'font-size': 11}, regime.short));
        observations.forEach((row, i) => {
          const jitter = ((((i * 17 + 5) % 23) / 22) - .5) * Math.min(29, inner / 9);
          const cx = x + jitter, cy = y(row[metric]);
          const attrs = {fill: regime.color, stroke: 'var(--sb-ink)', 'stroke-width': 1, tabindex: 0, role: 'button', 'data-city': row.target_city, 'aria-label': `${cityName(row.target_city)}, ${regime.name}, ${label} ${percent(row[metric])}. Select to compare this city across plots.`};
          let mark;
          if (regime.shape === 'square') mark = el('rect', {...attrs, x: cx - 3.7, y: cy - 3.7, width: 7.4, height: 7.4});
          else if (regime.shape === 'diamond') mark = el('path', {...attrs, d: `M${cx},${cy - 4.5}l4.5,4.5l-4.5,4.5l-4.5,-4.5Z`});
          else mark = el('circle', {...attrs, cx, cy, r: 3.7});
          const inspect = event => {
            showTooltip(row, metric, event, mark);
            if (selectedCity === null) highlightCity(row.target_city);
          };
          const select = () => {
            selectedCity = selectedCity === row.target_city ? null : row.target_city;
            pinnedPoint = selectedCity ? {row, metric} : null;
            highlightCity(selectedCity);
          };
          mark.addEventListener('pointerenter', inspect);
          mark.addEventListener('pointermove', event => showTooltip(row, metric, event, mark));
          mark.addEventListener('focus', inspect);
          mark.addEventListener('pointerleave', () => { hideTooltip(); resetHighlight(); });
          mark.addEventListener('blur', () => { hideTooltip(); resetHighlight(); });
          mark.addEventListener('click', event => { event.stopPropagation(); select(); });
          mark.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
          svg.append(mark);
          cityMarks.push(mark);
        });
      });
      svg.addEventListener('click', () => { selectedCity = null; pinnedPoint = null; highlightCity(null); });
    });
    highlightCity(selectedCity);
  }
  metrics.forEach(([metric, label]) => {
    const figure = document.createElement('figure');
    figure.className = 'sb-geo-figure';
    const heading = document.createElement('h4'); heading.textContent = label;
    const box = document.createElement('div'); box.className = 'sb-geo-chart';
    figure.append(heading, box); plotHost.append(figure);
    metricBoxes.push({box, metric, label});
  });
  new ResizeObserver(draw).observe(plotHost);
  draw();
})();
