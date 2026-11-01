"""Generate an inspectable Earth Engine Code Editor export script; never submit jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.config import ROOT


def export_script(source, years, radius, asset):
    if source not in ("dynamic_world", "alphaearth") or radius not in (0, 100, 300, 500):
        raise ValueError("Unsupported source or radius")
    collection = "GOOGLE/DYNAMICWORLD/V1" if source == "dynamic_world" else "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    bands = [f"A{i:02d}" for i in range(64)] if source == "alphaearth" else ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
    return f'''// Inspect in https://code.earthengine.google.com/; no automatic submission.
// Upload canonical_buildings.csv as an EE table with LONGITUDE/LATITUDE geometry.
// Annual observation years are NOT publication years. Add independently verified
// available_at and availability_source_url after export before importing locally.
var buildings = ee.FeatureCollection({json.dumps(asset)});
var collection = ee.ImageCollection({json.dumps(collection)});
var bands = {json.dumps(bands)};
print('Actual catalog observation years', collection.aggregate_array('system:time_start').map(function(t) {{return ee.Date(t).get('year');}}).distinct().sort());
{json.dumps(years)}.forEach(function(year) {{
  var images = collection.filterBounds(buildings).filterDate(ee.Date.fromYMD(year,1,1), ee.Date.fromYMD(year+1,1,1)).select(bands);
  // Dynamic World images are already cloud/shadow masked. Mean probabilities
  // retain class uncertainty; count/valid fraction quantify actual coverage.
  var image = {'images.mosaic()' if source == 'alphaearth' else 'images.mean()'};
  var count = images.select([bands[0]]).count().rename('valid_observation_count');
  var valid = image.select([bands[0]]).mask().unmask(0).rename('valid_pixel_fraction');
  var summary = image.addBands(count).addBands(valid);
  var regions = buildings.map(function(f) {{return ee.Feature({'f.geometry()' if radius == 0 else f'f.geometry().buffer({radius})'}, {{building_id:f.get('building_id')}});}});
  var features = summary.reduceRegions({{collection:regions, reducer:ee.Reducer.mean(), scale:10, crs:'EPSG:3414', tileScale:4}})
    .map(function(f) {{return f.set({{observation_year:year, radius_m:{radius}, observation_end:ee.Date.fromYMD(year+1,1,1).advance(-1,'day').format('YYYY-MM-dd')}}).setGeometry(null);}});
  Export.table.toDrive({{collection:features, description:'hdb_{source}_{radius}m_'+year, fileFormat:'CSV'}});
}});
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["dynamic_world", "alphaearth"], required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--radius", type=int, default=300)
    parser.add_argument("--building-asset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(export_script(args.source, args.years, args.radius, args.building_asset), encoding="utf-8")
    print(f"Wrote {args.output}. No cloud jobs submitted.")
