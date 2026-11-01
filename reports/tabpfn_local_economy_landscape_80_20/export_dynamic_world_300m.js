// Inspect in https://code.earthengine.google.com/; no automatic submission.
// Upload canonical_buildings.csv as an EE table with LONGITUDE/LATITUDE geometry.
// Annual observation years are NOT publication years. Add independently verified
// available_at and availability_source_url after export before importing locally.
var buildings = ee.FeatureCollection("users/YOUR_ACCOUNT/hdb_buildings");
var collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1");
var bands = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"];
print('Actual catalog observation years', collection.aggregate_array('system:time_start').map(function(t) {return ee.Date(t).get('year');}).distinct().sort());
[2020, 2021, 2022, 2023].forEach(function(year) {
  var images = collection.filterBounds(buildings).filterDate(ee.Date.fromYMD(year,1,1), ee.Date.fromYMD(year+1,1,1)).select(bands);
  // Dynamic World images are already cloud/shadow masked. Mean probabilities
  // retain class uncertainty; count/valid fraction quantify actual coverage.
  var image = images.mean();
  var count = images.select([bands[0]]).count().rename('valid_observation_count');
  var valid = image.select([bands[0]]).mask().unmask(0).rename('valid_pixel_fraction');
  var summary = image.addBands(count).addBands(valid);
  var regions = buildings.map(function(f) {return ee.Feature(f.geometry().buffer(300), {building_id:f.get('building_id')});});
  var features = summary.reduceRegions({collection:regions, reducer:ee.Reducer.mean(), scale:10, crs:'EPSG:3414', tileScale:4})
    .map(function(f) {return f.set({observation_year:year, radius_m:300, observation_end:ee.Date.fromYMD(year+1,1,1).advance(-1,'day').format('YYYY-MM-dd')}).setGeometry(null);});
  Export.table.toDrive({collection:features, description:'hdb_dynamic_world_300m_'+year, fileFormat:'CSV'});
});
