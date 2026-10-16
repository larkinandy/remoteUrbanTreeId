# Urban tree research storyboard

The Streamlit entry point serves the five-chapter narrative in `story.html`:
project purpose, datasets, discrimination-space groups, the two-stage model,
and final-model results and conclusion. Hovering over taxa updates their story;
the manuscript group legend highlights a group and fades the others to 10%.
The satellite figure appears inside the dataset detail panel.

## Run locally

From the repository root:

```powershell
python -m venv .venv-dashboard
.\.venv-dashboard\Scripts\python -m pip install -r storyboard/requirements.txt
.\.venv-dashboard\Scripts\python -m streamlit run storyboard/app.py
```

## Deploy to Streamlit Community Cloud

Select this repository and the branch containing these changes. Set the main
file path to `storyboard/app.py`. Dependencies are in
`storyboard/requirements.txt`. Commit and push `app.py`, `story.html`, and the
requirements before redeploying. Restart the existing app if it retains an
older deployment.

`story.html` is the deployable story source. It contains its images, coordinates,
manuscript colors, and interactions, so deployment does not require a local
Desktop directory, the Codex visualization directory, model weights, or a GPU.
Make future narrative and design edits in this file to update the deployed app.
The original dashboard's `data.py`, `panel.html`, and `interaction.js` remain as
reference code; the Streamlit entry point no longer renders that interface.

The displayed results use the repository's frozen artifacts. Overall scores
describe all 161,508 held-out records; group scores describe the 153,529 records
that passed the vegetation-height screen. The embedded image sequence comes
from Supplementary Figure S8.
