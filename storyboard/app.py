"""Run with: streamlit run storyboard/app.py"""
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from data import load_taxa

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Urban tree discrimination space", page_icon="🌳", layout="wide")
st.title("Urban tree discrimination space")
st.caption("Explore learned remote-sensing signatures and the hierarchy of taxon groups.")


@st.cache_data
def get_data():
    return load_taxa()


try:
    taxa, display = get_data()
except (ValueError, OSError) as error:
    st.error(f"Could not load validated figure data: {error}")
    st.stop()

with st.sidebar:
    st.header("Explore the space")
    level = st.radio("Group space", ["Collapsed k6", "Fine k11"])
    group_column = "collapsed_k6_group" if level == "Collapsed k6" else "manuscript_k11_group"
    palette = display["collapsed_k6_colors" if level == "Collapsed k6" else "k11_manuscript_colors"]
    groups = st.multiselect("Show groups", list(range(1, len(palette) + 1)), default=list(range(1, len(palette) + 1)))
    types = st.multiselect("Taxon provenance", ["core", "rare_extension"], default=["core", "rare_extension"])
    query = st.text_input("Search taxa", placeholder="e.g. Acer or Quercus")
    st.caption("Hover to inspect. Click a point to pin its details. Use the chart toolbar to zoom, pan, or reset.")

visible = taxa[taxa[group_column].isin(groups) & taxa.taxon_type.isin(types)]
if query.strip():
    visible = visible[visible.name.str.contains(query.strip().replace("_", " "), case=False, regex=False)]
left, center, right = st.columns(3)
left.metric("Taxa shown", f"{len(visible):,} / {len(taxa):,}")
center.metric("Core taxa shown", int((visible.taxon_type == "core").sum()))
right.metric("Rare-extension taxa shown", int((visible.taxon_type == "rare_extension").sum()))

if visible.empty:
    st.info("No taxa match these filters. Clear the search or select additional groups and provenance types.")
else:
    selected = st.selectbox("Inspect a taxon", visible.taxon.tolist(), format_func=lambda x: x.replace("_", " ").capitalize())
    fig = go.Figure()
    maximum = taxa.loc[taxa.taxon_type == "core", "records"].max()
    for group in groups:
        for kind in types:
            subset = visible[(visible[group_column] == group) & (visible.taxon_type == kind)]
            if subset.empty:
                continue
            sizes = (8 + 26 * (subset.records / maximum) ** .5).tolist() if kind == "core" else 9
            fig.add_trace(go.Scatter(
                x=subset.umap_1, y=subset.umap_2, mode="markers",
                name=f"Group {group} · {'core' if kind == 'core' else 'rare'}",
                customdata=subset.taxon.tolist(), text=subset.name.tolist(),
                marker=dict(size=sizes, color=palette[group - 1], symbol="circle" if kind == "core" else "cross",
                            opacity=.8, line=dict(width=1, color="white")),
                hovertemplate="<i>%{text}</i><extra></extra>",
            ))
    # Bounds remain fixed under filtering, as in the manuscript's shared space.
    def bounds(series):
        pad = (series.max() - series.min()) * .06
        return [series.min() - pad, series.max() + pad]
    fig.update_layout(height=650, margin=dict(l=45, r=15, t=25, b=45),
                      paper_bgcolor="white", plot_bgcolor="#f5f8f7", font=dict(family="Arial", color="#243c35"),
                      xaxis=dict(title="UMAP 1", range=bounds(taxa.umap_1), zeroline=False),
                      yaxis=dict(title="UMAP 2", range=bounds(taxa.umap_2), zeroline=False),
                      legend=dict(orientation="h", y=-.13), dragmode="pan")
    records = json.loads(taxa.to_json(orient="records"))
    script = (HERE / "interaction.js").read_text(encoding="utf-8")
    script = script.replace("__RECORDS__", json.dumps(records).replace("</", "<\\/"))
    script = script.replace("__SELECTED__", json.dumps(selected)).replace("__GROUP_COLUMN__", json.dumps(group_column))
    chart = pio.to_html(fig, full_html=False, include_plotlyjs=True, div_id="space",
                        config={"responsive": True, "displaylogo": False, "scrollZoom": True}, post_script=script)
    template = (HERE / "panel.html").read_text(encoding="utf-8")
    st.iframe(template.replace("<!--CHART-->", chart), height=850,
              alt="Interactive taxon discrimination space with live taxon and group details")
    st.download_button("Download visible taxa (CSV)", visible.to_csv(index=False), "discrimination_space_taxa.csv", "text/csv")

with st.expander("How to interpret this space"):
    st.markdown("""
    Each circle is a **core-taxon validation centroid** in a learned 64-dimensional
    embedding space, displayed using frozen UMAP coordinates. Circle sizes increase
    with validation support; crosses represent rare-extension taxa at a fixed size.

    Rare taxa were assigned using development-only centroids. Their displayed positions
    are weighted interpolations of neighboring core centroids, **not UMAP transforms**.
    Rare development and held-out counts are reported separately in the detail panel.

    Switching between k11 and k6 changes group colors, not positions. The k11 labels
    shown in the chart use manuscript display order; the detail panel also shows the
    original analytical model label. Group numbers do not indicate botanical ranks.
    UMAP axes have no physical units; apparent separation is not an accuracy estimate.
    Group summaries describe the full frozen dataset, independent of active filters.
    """)
