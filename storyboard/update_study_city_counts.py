"""Synchronize storyboard city counts with the final core-plus-rare split table."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent / "outputs/reviewer_revision/stage2_partitions_k11_groupnorm_v1"
    / "records_by_split_tier_city.csv"
)


def main():
    counts = Counter()
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            counts[row["city_token"], row["frequency_category"]] += int(row["records"])
    cities_path = HERE / "assets/study-cities.json"
    cities = json.loads(cities_path.read_text(encoding="utf-8"))
    assert {token for token, _ in counts} == {city["token"] for city in cities}
    for city in cities:
        token = city["token"]
        city["core_records"] = counts[token, "core_ge500"]
        city["rare_records"] = counts[token, "rare_extension_50_499"]
        city["records"] = city["core_records"] + city["rare_records"]
    assert sum(c["core_records"] for c in cities) == 1_794_468
    assert sum(c["rare_records"] for c in cities) == 57_500
    assert sum(c["records"] for c in cities) == 1_851_968

    story_path = HERE / "story.html"
    story = story_path.read_text(encoding="utf-8").replace(" (core + rare taxa)", "")
    embedded = re.search(r"const studyCities=(\[.*?\]);", story)
    if embedded is None:
        raise ValueError("Missing embedded studyCities data")
    payload = json.dumps(cities, separators=(",", ":"), ensure_ascii=False)
    story = story[:embedded.start(1)] + payload + story[embedded.end(1):]
    story = story.replace(
        "study records after final taxonomic exclusions`",
        "study records`",
    )
    svg_path = HERE / "assets/study-cities.svg"
    svg = svg_path.read_text(encoding="utf-8")
    for city in cities:
        pattern = (
            r'(<circle\b[^>]*data-study-city="' + re.escape(city["token"])
            + r'"[^>]*><title>)[^<]*(</title>)'
        )
        title = f'{city["city"]}, {city["state"]}: {city["records"]:,} trees'
        svg, n_svg = re.subn(pattern, lambda m: m[1] + title + m[2], svg)
        story, n_story = re.subn(pattern, lambda m: m[1] + title + m[2], story)
        assert n_svg == n_story == 1, (city["token"], n_svg, n_story)
    ET.fromstring(svg)
    cities_path.write_text(json.dumps(cities, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (HERE / "assets/study-city-filtered-records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city_token", "city", "state", "core_records", "rare_records", "filtered_records"])
        for c in cities:
            writer.writerow([c["token"], c["city"], c["state"], c["core_records"], c["rare_records"], c["records"]])
    svg_path.write_text(svg, encoding="utf-8")
    story_path.write_text(story, encoding="utf-8")
    print(f"Updated {len(cities)} cities: 1,794,468 core + 57,500 rare = 1,851,968 records")


if __name__ == "__main__":
    main()
