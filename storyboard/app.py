"""Serve the research storyboard: streamlit run storyboard/app.py."""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Urban tree research storyboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep the complete story in one frame so chapter links, hover details, and
# group highlighting share their state without rerunning Python.
story = (HERE / "story.html").read_text(encoding="utf-8")
st.iframe(
    story,
    height="content",
    alt="Urban tree research storyboard: purpose, datasets, discrimination groups, prediction, and results",
)
