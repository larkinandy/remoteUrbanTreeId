#!/usr/bin/env node
/** Render Sentinel-2 interpolation-gap figures from prepared CSV summaries. */

import fs from "node:fs";
import path from "node:path";

const INPUT_DIR = process.argv[2] ?? "outputs/manuscript_figures/sentinel_interpolation_gaps";
const BLUE = "#0b78b5";
const TEAL = "#3f8f83";
const DARK = "#25384c";
const MUTED = "#607284";
const GRID = "#d8dee5";
const ORANGE = "#d45a00";

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function parseCsv(filePath) {
  const [headerLine, ...lines] = fs.readFileSync(filePath, "utf8").trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function svgDocument(width, height, content) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<g font-family="Arial, Helvetica, sans-serif">${content}</g>
</svg>\n`;
}

function renderEventHistogram(rows) {
  const width = 1500;
  const height = 850;
  const margin = { left: 135, right: 45, top: 95, bottom: 165 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const minY = 1_000;
  const maxY = 200_000_000;
  const logMinY = Math.log10(minY);
  const logMaxY = Math.log10(maxY);
  const yScale = (value) => margin.top + plotHeight - ((Math.log10(Math.max(minY, value)) - logMinY) / (logMaxY - logMinY)) * plotHeight;
  const barSlot = plotWidth / rows.length;
  const barWidth = barSlot * 0.88;
  const elements = [];

  elements.push(`<text x="${margin.left}" y="48" fill="${DARK}" font-size="34" font-weight="700">Sentinel-2 interpolation-gap events</text>`);
  const yTicks = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
  const yTickLabels = ["1K", "10K", "100K", "1M", "10M", "100M"];
  yTicks.forEach((tick, tickIndex) => {
    const y = yScale(tick);
    elements.push(`<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${GRID}" stroke-width="1.5"/>`);
    elements.push(`<text x="${margin.left - 18}" y="${y + 8}" fill="${MUTED}" font-size="24" text-anchor="end">${yTickLabels[tickIndex]}</text>`);
  });
  const labeledBins = new Set([0, 1, ...Array.from({ length: rows.length }, (_, index) => index).filter((index) => index >= 3 && index % 3 === 0), rows.length - 1]);
  rows.forEach((row, index) => {
    const value = Number(row.gap_events);
    const x = margin.left + index * barSlot + (barSlot - barWidth) / 2;
    const y = yScale(value);
    const barHeight = margin.top + plotHeight - y;
    const fill = index === 0 ? TEAL : BLUE;
    elements.push(`<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${fill}"/>`);
    if (labeledBins.has(index)) {
      const labelX = x + barWidth / 2;
      const labelY = margin.top + plotHeight + 25;
      elements.push(`<text x="${labelX}" y="${labelY}" fill="${MUTED}" font-size="19" text-anchor="end" transform="rotate(-45 ${labelX} ${labelY})">${escapeXml(row.bin_label)}</text>`);
    }
  });
  elements.push(`<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#9aabba" stroke-width="2"/>`);
  elements.push(`<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#9aabba" stroke-width="2"/>`);
  elements.push(`<text x="${margin.left + plotWidth / 2}" y="${height - 35}" fill="${DARK}" font-size="27" text-anchor="middle">Interpolation-gap duration (days)</text>`);
  elements.push(`<text x="38" y="${margin.top + plotHeight / 2}" fill="${DARK}" font-size="27" text-anchor="middle" transform="rotate(-90 38 ${margin.top + plotHeight / 2})">Count (logarithmic scale)</text>`);
  elements.push(`<rect x="${margin.left}" y="68" width="19" height="19" fill="${TEAL}"/><text x="${margin.left + 29}" y="85" fill="${MUTED}" font-size="19">Observed intervals without interpolation</text>`);
  elements.push(`<rect x="${margin.left + 390}" y="68" width="19" height="19" fill="${BLUE}"/><text x="${margin.left + 419}" y="85" fill="${MUTED}" font-size="19">Interpolation-gap events</text>`);
  const noInterpolation = Number(rows[0].gap_events);
  const totalEvents = rows.slice(1).reduce((sum, row) => sum + Number(row.gap_events), 0);
  elements.push(`<text x="${width - margin.right}" y="66" fill="${MUTED}" font-size="20" text-anchor="end">${totalEvents.toLocaleString("en-US")} interpolation events</text>`);
  elements.push(`<text x="${width - margin.right}" y="92" fill="${MUTED}" font-size="20" text-anchor="end">${noInterpolation.toLocaleString("en-US")} observed intervals without interpolation</text>`);
  return svgDocument(width, height, elements.join("\n"));
}

function renderCityBoxplots(rows) {
  const width = 1500;
  const height = 1260;
  const margin = { left: 260, right: 60, top: 115, bottom: 110 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const minimumX = 5;
  const maximumX = 220;
  const logMin = Math.log10(minimumX);
  const logMax = Math.log10(maximumX);
  const xScale = (value) => margin.left + ((Math.log10(Math.max(minimumX, Number(value))) - logMin) / (logMax - logMin)) * plotWidth;
  const rowHeight = plotHeight / rows.length;
  const elements = [];
  const ticks = [5, 10, 20, 50, 100, 200];

  elements.push(`<text x="${margin.left}" y="49" fill="${DARK}" font-size="34" font-weight="700">Sentinel-2 interpolation gaps by city</text>`);
  elements.push(`<text x="${margin.left}" y="80" fill="${MUTED}" font-size="20">Longest gap per record; records without interpolation are shown at the expected 5-day interval</text>`);
  ticks.forEach((tick) => {
    const x = xScale(tick);
    elements.push(`<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}" stroke="${tick === 5 ? ORANGE : GRID}" stroke-width="${tick === 5 ? 2 : 1.2}" ${tick === 5 ? 'stroke-dasharray="8 6"' : ""}/>`);
    elements.push(`<text x="${x}" y="${margin.top + plotHeight + 34}" fill="${MUTED}" font-size="20" text-anchor="middle">${tick}</text>`);
  });

  rows.forEach((row, index) => {
    const y = margin.top + (index + 0.5) * rowHeight;
    const boxHeight = Math.min(19, rowHeight * 0.64);
    const low = xScale(row.lower_whisker);
    const q1 = xScale(row.q1);
    const median = xScale(row.median);
    const q3 = xScale(row.q3);
    const high = xScale(row.upper_whisker);
    elements.push(`<text x="${margin.left - 14}" y="${y + 6}" fill="${DARK}" font-size="18" text-anchor="end">${escapeXml(row.city_label)}</text>`);
    elements.push(`<line x1="${low}" y1="${y}" x2="${high}" y2="${y}" stroke="${DARK}" stroke-width="1.6"/>`);
    elements.push(`<line x1="${low}" y1="${y - boxHeight * 0.35}" x2="${low}" y2="${y + boxHeight * 0.35}" stroke="${DARK}" stroke-width="1.6"/>`);
    elements.push(`<line x1="${high}" y1="${y - boxHeight * 0.35}" x2="${high}" y2="${y + boxHeight * 0.35}" stroke="${DARK}" stroke-width="1.6"/>`);
    elements.push(`<rect x="${q1}" y="${y - boxHeight / 2}" width="${Math.max(1.5, q3 - q1)}" height="${boxHeight}" fill="${BLUE}" fill-opacity="0.82" stroke="${DARK}" stroke-width="1.3"/>`);
    elements.push(`<line x1="${median}" y1="${y - boxHeight / 2}" x2="${median}" y2="${y + boxHeight / 2}" stroke="#ffffff" stroke-width="2.6"/>`);
  });
  elements.push(`<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#9aabba" stroke-width="2"/>`);
  elements.push(`<text x="${margin.left + plotWidth / 2}" y="${height - 30}" fill="${DARK}" font-size="27" text-anchor="middle">Longest interpolation-gap duration per record (days, logarithmic scale)</text>`);
  elements.push(`<text x="${xScale(5) + 8}" y="${margin.top - 12}" fill="#a64b00" font-size="18">No interpolation (5 days)</text>`);
  return svgDocument(width, height, elements.join("\n"));
}

fs.mkdirSync(INPUT_DIR, { recursive: true });
const histogramRows = parseCsv(path.join(INPUT_DIR, "gap_event_histogram_data.csv"));
const cityRows = parseCsv(path.join(INPUT_DIR, "city_longest_gap_boxplot_summary.csv"));
fs.writeFileSync(path.join(INPUT_DIR, "Figure_S_sentinel_interpolation_gap_events.svg"), renderEventHistogram(histogramRows));
fs.writeFileSync(path.join(INPUT_DIR, "Figure_S_sentinel_longest_gap_by_city.svg"), renderCityBoxplots(cityRows));
