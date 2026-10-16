# Discrimination-space storyboard

From the repository root, install the lightweight dashboard environment and run:

```powershell
python -m venv .venv-dashboard
.\.venv-dashboard\Scripts\python -m pip install -r storyboard/requirements.txt
.\.venv-dashboard\Scripts\python -m streamlit run storyboard/app.py
```

No GPU, model weights, external services, or retraining are needed. Plotly's
JavaScript is embedded locally; the chart does not load assets from a CDN.

Hover over any point to update the adjacent taxon and group panel. Click to pin,
then use **Resume hover** to unpin. The taxon selector provides a keyboard-accessible
alternative. Filters and the selector rerender the chart and clear pinned state.
Use the toolbar to pan, zoom, reset, or save a PNG. Download the filtered table
using the CSV button.

The app reads frozen manuscript coordinates, display colors, and scientific
group mappings directly from the repository, validating taxon membership,
coordinates, support, and group correspondence before rendering. It distinguishes
manuscript and analytical k11 identifiers. Group summaries always use the full
dataset. Rare support is development plus held-out records, not validation support.

Live hover runs in an embedded Plotly panel so it does not rerun Python on each
mouse movement. Native Streamlit controls handle filters and downloads.
Source files in this directory belong in Git; virtual environments and caches
are already ignored by the repository's existing dot-directory/cache rules.
