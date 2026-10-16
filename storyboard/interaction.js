function initializeInspector() {
const records = __RECORDS__;
const byTaxon = Object.fromEntries(records.map(r => [r.taxon, r]));
const groupColumn = __GROUP_COLUMN__;
const chart = document.getElementById('space');
let pinned = false;
const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const count = value => value == null ? '—' : Number(value).toLocaleString();
function show(taxon) {
    const r = byTaxon[taxon];
    if (!r) return;
    const members = records.filter(x => x[groupColumn] === r[groupColumn]);
    const core = members.filter(x => x.taxon_type === 'core');
    const support = core.reduce((s,x) => s + x.core_validation_records, 0);
    const total = records.filter(x => x.taxon_type === 'core').reduce((s,x) => s+x.core_validation_records, 0);
    const field = (label,value) => `<dt>${label}</dt><dd>${escapeText(value)}</dd>`;
    document.getElementById('detail').innerHTML = `
      <h2>${escapeText(r.name)}</h2><span class="badge">${r.taxon_type === 'core' ? 'Core taxon' : 'Rare extension'}</span><dl>
      ${field('k11 · manuscript / analytical', r.manuscript_k11_group + ' / ' + r.k11_group)}
      ${field('Collapsed k6 group', r.collapsed_k6_group)}
      ${r.taxon_type === 'core' ? field('Validation records', count(r.core_validation_records)) :
        field('Development records',count(r.rare_development_records)) + field('Held-out test records',count(r.rare_held_out_test_records))}
      ${field('UMAP coordinates',r.umap_1.toFixed(3)+', '+r.umap_2.toFixed(3))}</dl>
      <h3>Group ${r[groupColumn]} · ${groupColumn === 'collapsed_k6_group' ? 'k6' : 'manuscript k11'}</h3>
      <p>${core.length} core + ${members.length-core.length} rare taxa<br>${count(support)} core validation records<br>${(100*support/total).toFixed(1)}% of core validation support</p>
      <p class="hint">All taxa in this group (full dataset)</p>
      <ul id="members">${members.map(x=>`<li>${escapeText(x.name)}</li>`).join('')}</ul>`;
}
chart.on('plotly_hover', event => {if (!pinned) show(event.points[0].customdata);});
chart.on('plotly_click', event => {
    pinned = true; show(event.points[0].customdata);
    document.getElementById('status').textContent = 'Pinned · click another taxon or resume hover';
});
document.getElementById('unpin').onclick = () => {
    pinned = false;
    document.getElementById('status').textContent = 'Hover to inspect · click to pin';
};
show(__SELECTED__);
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeInspector, {once: true});
} else {
    initializeInspector();
}
