"""Embed the Denver raw-return crop beside its matching CHM in the LiDAR tab."""
import base64
import io
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
from draw_storyboard_raw_lidar_returns import colors
import numpy as np
from PIL import Image, ImageDraw


def main():
    here = Path(__file__).resolve().parent
    cache = ROOT / "outputs/manuscript_supplement/raw_lidar_denver_tree_52199.npz"
    with np.load(cache) as data:
        x, y, z = data["x"], data["y"], data["z"]
    size = 600
    image = Image.new("RGB", (size, size), "#F1F3F5")
    draw = ImageDraw.Draw(image)
    rgb = colors(z, float(z.min()), float(z.max()))
    for i in np.argsort(z):
        px = int((x[i] + 19) / 38 * (size - 1))
        py = int((19 - y[i]) / 38 * (size - 1))
        draw.ellipse((px-1, py-1, px+1, py+1), fill=tuple(rgb[i]))
    asset = here / "assets/denver-red-oak-raw-lidar-returns.png"
    image.save(asset)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True)
    uri = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    path = here / "story.html"
    story = path.read_text(encoding="utf-8")
    pattern = r'<img\b[^>]*alt="Aerial view of a northern red oak crown"[^>]*>'
    match = re.search(pattern, story)
    if not match:
        raise ValueError("Cannot find the first aerial-image panel")
    tag = match[0]
    original = re.search(r'data-aerial-src="([^"]+)"', tag)
    original = original[1] if original else re.search(r'\bsrc="([^"]+)"', tag)[1]
    tag = re.sub(r' data-(?:aerial|raw-lidar)-src="[^"]*"', '', tag)
    tag = tag[:-1] + f' data-aerial-src="{original}" data-raw-lidar-src="{uri}">'
    story = story[:match.start()] + tag + story[match.end():]

    anchor = "const chip=root.querySelector('.sb-chip-pair figure:nth-child(2) img');"
    replacement = (
        "const first=root.querySelector('.sb-chip-pair figure:first-child img');"
        "const isLidar=k==='lidar';"
        "first.src=isLidar?first.dataset.rawLidarSrc:first.dataset.aerialSrc;"
        "first.alt=isLidar?'Raw LiDAR returns for the same northern red oak in Denver, tree_id 52199':'Aerial view of a northern red oak crown';"
        "root.querySelector('.sb-chip-pair figure:first-child figcaption').textContent=isLidar?'Raw LiDAR returns':'Aerial imagery';"
        + anchor
    )
    if story.count(anchor) != 1:
        raise ValueError("Expected one image-switching anchor")
    if "const isLidar=k==='lidar';" not in story:
        story = story.replace(anchor, replacement)
    path.write_text(story, encoding="utf-8")
    print(f"Updated {path}; embedded {len(z):,} raw returns for Denver tree_id 52199")


if __name__ == "__main__":
    main()
