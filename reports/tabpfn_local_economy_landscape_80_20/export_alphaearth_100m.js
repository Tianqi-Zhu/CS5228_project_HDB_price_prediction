// Inspect in https://code.earthengine.google.com/; no automatic submission.
// Upload canonical_buildings.csv as an EE table with LONGITUDE/LATITUDE geometry.
// Annual observation years are NOT publication years. Add independently verified
// available_at and availability_source_url after export before importing locally.
var buildings = ee.FeatureCollection("users/YOUR_ACCOUNT/hdb_buildings");
var collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL");
var bands = ["A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A50", "A51", "A52", "A53", "A54", "A55", "A56", "A57", "A58", "A59", "A60", "A61", "A62", "A63"];
print('Actual catalog observation years', collection.aggregate_array('system:time_start').map(function(t) {return ee.Date(t).get('year');}).distinct().sort());
[2020, 2021, 2022, 2023].forEach(function(year) {
  var images = collection.filterBounds(buildings).filterDate(ee.Date.fromYMD(year,1,1), ee.Date.fromYMD(year+1,1,1)).select(bands);
  // Dynamic World images are already cloud/shadow masked. Mean probabilities
  // retain class uncertainty; count/valid fraction quantify actual coverage.
  var image = images.mosaic();
  var count = images.select([bands[0]]).count().rename('valid_observation_count');
  var valid = image.select([bands[0]]).mask().unmask(0).rename('valid_pixel_fraction');
  var summary = image.addBands(count).addBands(valid);
  var regions = buildings.map(function(f) {return ee.Feature(f.geometry().buffer(100), {building_id:f.get('building_id')});});
  var features = summary.reduceRegions({collection:regions, reducer:ee.Reducer.mean(), scale:10, crs:'EPSG:3414', tileScale:4})
    .map(function(f) {return f.set({observation_year:year, radius_m:100, observation_end:ee.Date.fromYMD(year+1,1,1).advance(-1,'day').format('YYYY-MM-dd')}).setGeometry(null);});
  Export.table.toDrive({collection:features, description:'hdb_alphaearth_100m_'+year, fileFormat:'CSV'});
});
