"""Render all daily PRISM observations for the storyboard's Anaheim tree."""
from pathlib import Path
import base64
from datetime import datetime
import io
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = Path(r"H:\TreeCenteredModelInputs\tree_centered_complete_sharded100k_prism_daily3_clean\anaheim")
PRISM = Path(r"H:\tree_centered_prism_daily_clean")


def font(size):
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)


def main():
    metadata = pd.read_csv(next(DATA.glob("*metadata.csv")), usecols=["tree_id", "scientific_name"])
    positions = np.flatnonzero(metadata.tree_id.to_numpy() == 63599)
    assert len(positions) == 1
    i = positions[0]
    assert metadata.iloc[i].scientific_name.strip() == "Auranticarpa rhombifolia"
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D").strftime("%Y%m%d").to_numpy(dtype=np.int32)
    with np.load(PRISM / "anaheim/anaheim_prism_daily_index.npz", allow_pickle=False) as index:
        positions = np.flatnonzero(index["tree_id"] == 63599)
        assert len(positions) == 1
        j = positions[0]
        assert not index["missing_prism_daily"][j]
        assert int(index["prism_grid_row"][j]) == 386
        assert int(index["prism_grid_col"][j]) == 169
        source_row = int(index["prism_daily_source_row"][j])
    with np.load(PRISM / "prism_daily_values.npz", allow_pickle=False) as daily:
        all_dates = daily["dates_yyyymmdd"]
        date_positions = np.searchsorted(all_dates, dates)
        assert np.all(date_positions < len(all_dates))
        assert np.array_equal(all_dates[date_positions], dates)
        assert int(daily["prism_grid_row"][source_row]) == 386
        assert int(daily["prism_grid_col"][source_row]) == 169
        columns = list(daily["variable_names"])
        assert daily["valid_mask"][source_row, date_positions].all()
        values = daily["values"][source_row, date_positions]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "date", pd.to_datetime(dates.astype(str), format="%Y%m%d"))
    assert np.isfinite(frame[columns].to_numpy()).all()
    for _, group in frame.groupby("date"):
        assert np.allclose(group[columns], group[columns].iloc[0])
    frame = frame.drop_duplicates("date").sort_values("date")
    assert len(frame) == 1095
    frame.to_csv(HERE / "assets/anaheim-tree-63599-prism-daily.csv", index=False)

    width, height = 1440, 1020
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    left, right = 140, width-90
    start, end = datetime(2021, 1, 1), datetime(2024, 1, 1)
    def xp(date):
        return left + (date-start).total_seconds() / (end-start).total_seconds() * (right-left)
    ticks = [datetime(year,1,1) for year in range(2021,2025)]
    variables = [
        ("ppt", "Precipitation (mm)", "#2563A0"),
        ("tmean", "Mean temperature (\u00b0C)", "#BD612B"),
        ("vpdmax", "Maximum vapor-pressure deficit (hPa)", "#35785E"),
    ]
    for row,(key,label,color) in enumerate(variables):
        top = 70 + row*305
        bottom = top + 225
        draw.text((left,top-45),label,font=font(29),fill="#14213D")
        data=frame[key].to_numpy()
        lo=0 if key!='tmean' else math.floor(min(0,float(data.min()))/5)*5
        step=5 if key=='tmean' else 10 if key=='vpdmax' else max(1,math.ceil(float(data.max())/4))
        hi=max(lo+step,math.ceil(float(data.max())/step)*step)
        if key == 'ppt':
            hi = 10
        def yp(v):return bottom-(v-lo)/(hi-lo)*(bottom-top)
        for value in (np.arange(0,11,2) if key == 'ppt' else np.linspace(lo,hi,5)):
            y=yp(value)
            draw.line((left,y,right,y),fill="#E2E7EC",width=2)
            draw.text((left-18,y),f"{value:g}",font=font(23),fill="#526174",anchor="rm")
        for tick in ticks:
            x=xp(tick)
            draw.line((x,top,x,bottom),fill="#F0F2F4",width=1)
            if row==2:
                draw.text((x,bottom+20),tick.strftime('%b %Y'),font=font(23),fill="#526174",anchor="mt")
        points=[(xp(date.to_pydatetime()),yp(min(float(value),hi))) for date,value in zip(frame.date,data)]
        if key=='ppt':
            for x,y in points:draw.line((x,bottom,x,y),fill=color,width=4)
        else:
            draw.line(points,fill=color,width=4)
        for x,y in points:draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
        if key == 'ppt':
            for date,value in zip(frame.date,data):
                if value > hi:
                    x=xp(date.to_pydatetime())
                    draw.rectangle((x-10,top-12,x+10,top+12),fill='white')
                    draw.line((x-8,top,x+8,top),fill=color,width=4)
                    draw.line((x,top-8,x,top+8),fill=color,width=4)
        draw.line((left,top,left,bottom,right,bottom),fill="#94A3B8",width=2)
    asset=HERE/'assets/anaheim-tree-63599-prism-daily.png'
    canvas.save(asset)
    buffer=io.BytesIO();canvas.save(buffer,format='WEBP',lossless=True)
    uri='data:image/webp;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
    markup=(f'<figure class="sb-weather-example" hidden aria-label="Daily PRISM weather, 2021 to 2023">'
            f'<img src="{uri}" alt="Daily precipitation, mean temperature, and maximum vapor-pressure deficit from 2021 through 2023 for Anaheim tree 63599"></figure>')
    path=HERE/'story.html';story=path.read_text(encoding='utf-8')
    anchor=story.index('<figure class="sb-satellite-example"')
    stop=story.index('</figure>',anchor)+len('</figure>')
    if '<figure class="sb-weather-example"' in story:
        a=story.index('<figure class="sb-weather-example"');b=story.index('</figure>',a)+len('</figure>')
        story=story[:a]+markup+story[b:]
    else:story=story[:stop]+markup+story[stop:]
    old="root.querySelector('.sb-crown-examples').hidden=k==='sentinel'||k==='inventory';"
    new="root.querySelector('.sb-crown-examples').hidden=k==='sentinel'||k==='inventory'||k==='weather';"
    story=story.replace(old,new)
    toggle="root.querySelector('.sb-weather-example').hidden=k!=='weather';"
    if toggle not in story:
        story=story.replace("root.querySelector('.sb-satellite-example').hidden=k!=='sentinel';", "root.querySelector('.sb-satellite-example').hidden=k!=='sentinel';"+toggle)
    style='<style>#urban-forest-storyboard .sb-weather-example{margin:0}#urban-forest-storyboard .sb-weather-example img{display:block;width:100%;height:auto}</style>'
    if style not in story:story+= '\n'+style+'\n'
    path.write_text(story,encoding='utf-8')
    print(f'Updated weather tab: tree_id=63599; daily observations={len(frame)}; {frame.date.min().date()} to {frame.date.max().date()}')
    print(asset)


if __name__=='__main__':main()
