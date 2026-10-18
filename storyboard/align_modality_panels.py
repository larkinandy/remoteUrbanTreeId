"""Apply one shared image-pair layout to the aerial and LiDAR dataset views."""
from pathlib import Path

path = Path(__file__).resolve().with_name("story.html")
story = path.read_text(encoding="utf-8")
old = "#urban-forest-storyboard .sb-chip-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}#urban-forest-storyboard .sb-chip-pair img{width:100%;display:block;aspect-ratio:1;object-fit:cover}"
new = "#urban-forest-storyboard .sb-chip-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:14px;align-items:start}#urban-forest-storyboard .sb-chip-pair figure{min-width:0;margin:0}#urban-forest-storyboard .sb-chip-pair img{width:100%;height:auto;display:block;aspect-ratio:1;object-fit:cover}#urban-forest-storyboard .sb-chip-pair figcaption{margin:10px 0 0;line-height:1.4;min-height:2.8em}"
if old in story:
    story = story.replace(old,new)
elif new not in story:
    raise ValueError("Expected shared panel layout not found")
path.write_text(story,encoding="utf-8")
print("Updated shared panel spacing; content-specific captions retained.")
