"""
data_preparation.py — London Road Safety Data Pipeline
=======================================================
Combines all preprocessing steps (STATS19 loading, GLA filtering, IMD joining,
vehicle/casualty aggregation, missing-value treatment, temporal features,
OSMnx graph construction & centrality, AADF traffic counts, LAEI speeds,
and feature engineering) into a single reproducible script.

Outputs
-------
  data/gdf_prepared.geoparquet   — collision-level GeoDataFrame (EPSG:4326)
  data/casualties_ldn.parquet    — raw London casualty rows (for EDA)

Cached intermediates (created on first run, reused thereafter)
--------------------------------------------------------------
  data/collisions.parquet, vehicles.parquet, casualties.parquet
  data/london_centrality.parquet
  data/count_points.parquet, dft_traffic_counts_aadf.parquet
  data/laei2022_major_roads_speeds.parquet
  data/laei2022_minor_roads_speed_grid.parquet
  large/london_graph.graphml

Usage
-----
  cd CASA_DSSS_Assessment
  python data_preparation.py
"""

import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
from pathlib import Path
import urllib.request
import igraph as ig
import time
import zipfile
import urllib.request
import fiona
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

RAW_DIR = Path('large')
CACHE_DIR = Path('data')
RAW_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# 3.1  Load STATS19 Tables
# =============================================================================
TABLES = {
    'collisions': {
        'url':     ('https://data.dft.gov.uk/road-accidents-safety-data/'
                    'dft-road-casualty-statistics-collision-last-5-years.csv'),
        'raw':     RAW_DIR / 'dft-road-casualty-statistics-collision-last-5-years.csv',
        'parquet': CACHE_DIR / 'collisions.parquet',
    },
    'vehicles': {
        'url':     ('https://data.dft.gov.uk/road-accidents-safety-data/'
                    'dft-road-casualty-statistics-vehicle-last-5-years.csv'),
        'raw':     RAW_DIR / 'dft-road-casualty-statistics-vehicle-last-5-years.csv',
        'parquet': CACHE_DIR / 'vehicles.parquet',
    },
    'casualties': {
        'url':     ('https://data.dft.gov.uk/road-accidents-safety-data/'
                    'dft-road-casualty-statistics-casualty-last-5-years.csv'),
        'raw':     RAW_DIR / 'dft-road-casualty-statistics-casualty-last-5-years.csv',
        'parquet': CACHE_DIR / 'casualties.parquet',
    },
}


def load_table(name: str, cfg: dict) -> pd.DataFrame:
    """Load a STATS19 table: parquet cache -> raw CSV -> download CSV."""
    if cfg['parquet'].exists():
        print(f"[{name}] Loading from parquet cache: {cfg['parquet']}")
        df = pd.read_parquet(cfg['parquet'])
        print(f"[{name}] {len(df):,} rows x {df.shape[1]} cols")
        return df
    if cfg['raw'].exists():
        print(f"[{name}] Parquet not found. Reading raw CSV: {cfg['raw']}")
    else:
        print(f"[{name}] Not found locally. Downloading from DfT...")
        urllib.request.urlretrieve(cfg['url'], cfg['raw'])
        print(f"[{name}] Download complete: {cfg['raw'].stat().st_size / 1e6:.1f} MB")
    df = pd.read_csv(cfg['raw'], low_memory=False)
    print(f"[{name}] {len(df):,} rows x {df.shape[1]} cols")
    df.to_parquet(cfg['parquet'], index=False)
    return df


print("=" * 70)
print("  STAGE 1: Loading STATS19 tables")
print("=" * 70)
collisions = load_table('collisions', TABLES['collisions'])
vehicles = load_table('vehicles',   TABLES['vehicles'])
casualties = load_table('casualties', TABLES['casualties'])
print(f"\nYears covered: {sorted(collisions['collision_year'].unique())}")


# =============================================================================
# 3.2  Filter to Greater London
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 2: Filtering to Greater London boundary")
print("=" * 70)

GLA_SHP = Path('data/London_GLA_Boundary.shp')
gla = gpd.read_file(GLA_SHP)
if gla.crs.to_epsg() != 4326:
    gla = gla.to_crs(epsg=4326)
gla_boundary = gla.dissolve().geometry.iloc[0]

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = (
    gla_boundary.bounds[0] - 0.01,
    gla_boundary.bounds[2] + 0.01,
    gla_boundary.bounds[1] - 0.01,
    gla_boundary.bounds[3] + 0.01,
)

bbox_mask = (
    collisions['longitude'].between(LON_MIN, LON_MAX) &
    collisions['latitude'].between(LAT_MIN, LAT_MAX)
)
collisions_bbox = collisions[bbox_mask].copy()
print(f"Bounding box pre-filter: {len(collisions_bbox):,} candidates "
      f"(from {len(collisions):,} national)")

collisions_gdf = gpd.GeoDataFrame(
    collisions_bbox,
    geometry=gpd.points_from_xy(
        collisions_bbox['longitude'], collisions_bbox['latitude']
    ),
    crs='EPSG:4326'
)
collisions_gdf = collisions_gdf.dropna(subset=['longitude', 'latitude'])
london = collisions_gdf[collisions_gdf.geometry.within(gla_boundary)].copy()
print(f"London collisions (GLA boundary): {len(london):,}")

london_idx = set(london['collision_index'])
vehicles_ldn = vehicles[vehicles['collision_index'].isin(london_idx)].copy()
casualties_ldn = casualties[casualties['collision_index'].isin(
    london_idx)].copy()
print(f"London vehicles:   {len(vehicles_ldn):,}")
print(f"London casualties: {len(casualties_ldn):,}")


# =============================================================================
# 3.3  Clean & Preprocess Collision Table
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 3: Cleaning collision table")
print("=" * 70)

london['severity_binary'] = (london['collision_severity'] <= 2).astype(int)
print(
    f"Class balance:\n{london['severity_binary'].value_counts(normalize=True).round(3)}")

collision_features = [
    'road_type', 'first_road_class', 'speed_limit', 'junction_detail',
    'junction_control', 'light_conditions', 'weather_conditions',
    'road_surface_conditions', 'urban_or_rural_area',
    'pedestrian_crossing', 'special_conditions_at_site',
    'carriageway_hazards', 'number_of_vehicles', 'day_of_week'
]

london[collision_features] = london[collision_features].replace(
    [-1, 9], np.nan)
london_clean = london.dropna(subset=['latitude', 'longitude']).copy()
print(f"After coordinate cleaning: {len(london_clean):,} records")


# =============================================================================
# 3.4  Load & Join IMD Data
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 4: Loading and joining IMD 2019")
print("=" * 70)

imd = pd.read_excel('data/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx',
                    sheet_name='IMD2019')
imd = imd.rename(columns={
    'LSOA code (2011)':                          'lsoa_code',
    'Index of Multiple Deprivation (IMD) Decile': 'imd_decile',
    'Index of Multiple Deprivation (IMD) Rank':  'imd_rank',
})

LSOA_LOOKUP_CACHE = Path('data/lsoa_2011_to_2021_lookup.parquet')
BASE_URL = (
    'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/'
    'LSOA11_LSOA21_LAD22_EW_LU_v5/FeatureServer/0/query')

if not LSOA_LOOKUP_CACHE.exists():
    import requests
    records, offset, page_size = [], 0, 1000
    print("Downloading LSOA lookup (paginated)...")
    while True:
        params = {
            'where': '1=1', 'outFields': 'LSOA11CD,LSOA21CD,CHGIND',
            'f': 'json', 'resultOffset': offset, 'resultRecordCount': page_size,
        }
        data = requests.get(BASE_URL, params=params, timeout=30).json()
        features = data.get('features', [])
        records.extend([f['attributes'] for f in features])
        offset += len(features)
        if len(features) < page_size:
            break
    lsoa_lookup = (
        pd.DataFrame(records)[['LSOA11CD', 'LSOA21CD', 'CHGIND']]
        .drop_duplicates()
    )
    lsoa_lookup.to_parquet(LSOA_LOOKUP_CACHE, index=False)
    print(f"Downloaded {len(lsoa_lookup):,} rows -> cached")
else:
    lsoa_lookup = pd.read_parquet(LSOA_LOOKUP_CACHE)

lsoa21_to_11 = (lsoa_lookup.drop_duplicates('LSOA21CD', keep='first')
                .set_index('LSOA21CD')['LSOA11CD'].to_dict())
lsoa11_to_21 = (lsoa_lookup.drop_duplicates('LSOA11CD', keep='first')
                .set_index('LSOA11CD')['LSOA21CD'].to_dict())

imd['LSOA21CD'] = imd['lsoa_code'].map(lsoa11_to_21)
print(f"LSOA lookup: {len(lsoa_lookup):,} rows loaded")
print(f"IMD LSOA21CD coverage: {imd['LSOA21CD'].notna().mean():.1%}")


# =============================================================================
# 3.5  Aggregate Vehicle Table to Collision Level
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 5: Vehicle aggregation")
print("=" * 70)

vehicles_ldn = vehicles_ldn.replace(-1, np.nan)

MOTORCYCLE_TYPES = [2, 3, 4, 5, 97]
HGV_TYPES = [20, 21, 90, 98]
CYCLE_TYPES = [1]

vehicle_agg = vehicles_ldn.groupby('collision_index').agg(
    motorcycle_involved=(
        'vehicle_type', lambda x: x.isin(MOTORCYCLE_TYPES).any()),
    hgv_involved=('vehicle_type', lambda x: x.isin(HGV_TYPES).any()),
    cycle_involved=('vehicle_type', lambda x: x.isin(CYCLE_TYPES).any()),
    any_skid_overturn=('skidding_and_overturning',
                       lambda x: x.isin([1, 2, 3, 4, 5]).any()),
    any_left_carriageway=('vehicle_leaving_carriageway',
                          lambda x: x.isin([1, 2, 3, 4, 5, 6, 7, 8]).any()),
    any_hit_offroad_object=('hit_object_off_carriageway', lambda x: x.isin(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).any()),
    any_front_impact=('first_point_of_impact', lambda x: (x == 1).any()),
    any_moving_vehicle=('vehicle_manoeuvre', lambda x: x.isin(
        [1, 4, 5, 11, 12, 13, 14, 15, 19]).any()),
    any_turning_vehicle=('vehicle_manoeuvre',
                         lambda x: x.isin([6, 7, 8, 9, 10]).any()),
    mean_driver_age=('age_of_driver', 'mean'),
    min_driver_age=('age_of_driver', 'min'),
    work_journey=('journey_purpose_of_driver', lambda x: x.isin([1, 2]).any()),
    emergency_vehicle=('journey_purpose_of_driver', lambda x: (x == 8).any()),
    any_male_driver=('sex_of_driver', lambda x: (x == 1).any()),
    mean_vehicle_age=('age_of_vehicle', 'mean'),
    electric_involved=('propulsion_code', lambda x: x.isin([3, 8, 12]).any()),
    mean_driver_imd=('driver_imd_decile', 'mean'),
    escooter_involved=('escooter_flag', lambda x: (x == 1).any()),
).reset_index()

bool_cols = [
    'motorcycle_involved', 'hgv_involved', 'cycle_involved',
    'any_skid_overturn', 'any_left_carriageway', 'any_hit_offroad_object',
    'any_front_impact', 'any_moving_vehicle', 'any_turning_vehicle', 'any_male_driver',
    'work_journey', 'emergency_vehicle', 'electric_involved', 'escooter_involved'
]
vehicle_agg[bool_cols] = vehicle_agg[bool_cols].astype(int)
print(f"Vehicle aggregation: {len(vehicle_agg):,} rows")


# =============================================================================
# 3.6  Aggregate Casualty Table to Collision Level
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 6: Casualty aggregation")
print("=" * 70)

casualties_ldn = casualties_ldn.replace(-1, np.nan)

casualty_agg = casualties_ldn.groupby('collision_index').agg(
    n_casualties=('casualty_reference', 'count'),
    pedestrian_casualty=('casualty_type', lambda x: (x == 0).any()),
    cyclist_casualty=('casualty_type', lambda x: (x == 1).any()),
    motorcyclist_casualty=(
        'casualty_type', lambda x: x.isin([2, 3, 4, 5, 23, 97]).any()),
    child_casualty=('age_of_casualty', lambda x: (x < 16).any()),
    elderly_casualty=('age_of_casualty', lambda x: (x >= 70).any()),
    min_casualty_age=('age_of_casualty', 'min'),
    mean_casualty_age=('age_of_casualty', 'mean'),
    ped_on_crossing=('pedestrian_location', lambda x: x.isin([1, 2, 3]).any()),
    ped_jaywalking=('pedestrian_location',
                    lambda x: x.isin([4, 5, 6, 7, 8, 9]).any()),
    mean_casualty_imd=('casualty_imd_decile', 'mean'),
).reset_index()

bool_cols_c = ['pedestrian_casualty', 'cyclist_casualty', 'motorcyclist_casualty',
               'child_casualty', 'elderly_casualty', 'ped_on_crossing', 'ped_jaywalking']
casualty_agg[bool_cols_c] = casualty_agg[bool_cols_c].astype(int)
print(f"Casualty aggregation: {len(casualty_agg):,} rows")


# =============================================================================
# 3.7  Join All Tables Together
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 7: Joining tables")
print("=" * 70)

gdf = gpd.GeoDataFrame(
    london_clean,
    geometry=gpd.points_from_xy(london_clean.longitude, london_clean.latitude),
    crs='EPSG:4326'
)
gdf = gdf.merge(vehicle_agg,  on='collision_index', how='left')
gdf = gdf.merge(casualty_agg, on='collision_index', how='left')

# Join area-level IMD via LSOA
gdf = gdf.merge(
    imd[['lsoa_code', 'imd_decile', 'imd_rank']],
    left_on='lsoa_of_accident_location', right_on='lsoa_code', how='left'
)

imd_lookup = imd.set_index('lsoa_code')[['imd_decile', 'imd_rank']]
missing = gdf['imd_decile'].isna()
lsoa11_mapped = gdf.loc[missing, 'lsoa_of_accident_location'].map(lsoa21_to_11)
gdf.loc[missing, 'imd_decile'] = lsoa11_mapped.map(imd_lookup['imd_decile'])
gdf.loc[missing, 'imd_rank'] = lsoa11_mapped.map(imd_lookup['imd_rank'])
gdf['lsoa_code'] = gdf['lsoa_code'].fillna(gdf['lsoa_of_accident_location'])

print(f"Joined dataset: {len(gdf):,} rows, {gdf.shape[1]} columns")
print(f"IMD coverage: {gdf['imd_decile'].notna().mean():.1%}")


# =============================================================================
# 3.8a  Diagnosing Missing Values
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 8: Missing value treatment")
print("=" * 70)

null_summary = (
    gdf.isnull().sum()
    .pipe(lambda s: s[s > 0])
    .sort_values(ascending=False)
    .reset_index()
)
null_summary.columns = ['column', 'null_count']
null_summary['null_pct'] = (
    null_summary['null_count'] / len(gdf) * 100).round(1)
print(f"Columns with nulls: {len(null_summary)} of {gdf.shape[1]}")

# 3.8b  Treating Missing Values
gdf['junction_control'] = gdf['junction_control'].fillna(-1)

median_cols = [
    'mean_driver_age', 'min_driver_age', 'mean_driver_imd',
    'mean_casualty_imd', 'mean_vehicle_age'
]
for col in median_cols:
    median_val = gdf[col].median()
    n_filled = gdf[col].isna().sum()
    gdf[col] = gdf[col].fillna(median_val)
    print(f"Median filled: {col} ({n_filled:,} nulls -> {median_val:.1f})")

modal_cols = [
    'road_type', 'junction_detail', 'special_conditions_at_site',
    'weather_conditions', 'road_surface_conditions',
]
for col in modal_cols:
    mode_val = gdf[col].mode()[0]
    n_filled = gdf[col].isna().sum()
    gdf[col] = gdf[col].fillna(mode_val)
    print(f"Modal filled:  {col} ({n_filled:,} nulls -> {mode_val})")

drop_null_cols = [
    'min_casualty_age', 'mean_casualty_age',
    'number_of_vehicles', 'speed_limit',
    'pedestrian_crossing', 'carriageway_hazards'
]
before = len(gdf)
gdf = gdf.dropna(subset=drop_null_cols)
dropped = before - len(gdf)
print(f"Dropped {dropped:,} rows with nulls in essential columns "
      f"({dropped/before:.2%} of dataset)")
print(f"Cleaned dataset: {len(gdf):,} rows x {gdf.shape[1]} columns")


# =============================================================================
# 3.9  Temporal Feature Extraction
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 9: Temporal features")
print("=" * 70)

# --- 1. Existing Time Logic ---
gdf['hour_of_collision'] = pd.to_datetime(
    gdf['time'], format='%H:%M', errors='coerce'
).dt.hour


def assign_time_period(hour):
    if pd.isna(hour):
        return 'unknown'
    if hour < 4:
        return 'night'
    elif hour < 7:
        return 'early_am'
    elif hour < 9:
        return 'am_peak'
    elif hour < 12:
        return 'mid_morning'
    elif hour < 14:
        return 'lunchtime'
    elif hour < 16:
        return 'afternoon'
    elif hour < 19:
        return 'pm_peak'
    elif hour < 22:
        return 'evening'
    else:
        return 'night'


gdf['time_period'] = gdf['hour_of_collision'].apply(assign_time_period)

TIME_PERIOD_ORDER = ['afternoon', 'lunchtime', 'mid_morning', 'pm_peak',
                     'am_peak', 'evening', 'night', 'early_am', 'unknown']
tp_map = {p: i for i, p in enumerate(TIME_PERIOD_ORDER)}
gdf['time_period_enc'] = gdf['time_period'].map(tp_map)

# --- 2. New Year & Quarterly Logic ---
gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce')
gdf['year'] = gdf['date'].dt.year
month = gdf['date'].dt.month


def assign_quarter(m):
    if pd.isna(m):
        return None
    if 1 <= m <= 3:
        return 1   # Q1: Jan-Mar
    elif 4 <= m <= 6:
        return 2  # Q2: Apr-Jun
    elif 7 <= m <= 9:
        return 3  # Q3: Jul-Sep
    else:
        return 4              # Q4: Oct-Dec

gdf['quarter'] = month.apply(assign_quarter)

# Keep 'seasonal' for backward compatibility, but we'll use 'quarter' for modeling
gdf['seasonal'] = gdf['quarter']

# --- 3. Weekend/Special Logic ---
gdf['weekend_night'] = (
    (gdf['day_of_week'].isin([1, 7])) &
    (gdf['time_period'].isin(['night', 'early_am']))
).astype(int)

gdf['sun_early_am'] = (
    (gdf['day_of_week'] == 1) &
    (gdf['time_period'] == 'early_am')
).astype(int)

# --- 4. Print Summary ---
print(
    f"  hour_of_collision : {gdf['hour_of_collision'].notna().sum():,} non-null")
print(f"  year              : {gdf['year'].notna().sum():,} non-null")
print(f"  quarter (seasonal): {gdf['quarter'].notna().sum():,} non-null")
print(
    f"  weekend_night     : {gdf['weekend_night'].sum():,} ({gdf['weekend_night'].mean():.1%})")

# =============================================================================
# 3.10a  Graph Construction (OSMnx)
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 10: OSMnx graph + centrality")
print("=" * 70)


GRAPH_CACHE = Path('large/london_graph.graphml')
CENTRALITY_CACHE = Path('data/london_centrality.parquet')

if GRAPH_CACHE.exists():
    print("Loading cached graph...")
    G = ox.load_graphml(GRAPH_CACHE)
    print(f"Cached graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
else:
    print("Downloading London road network from OSM...")
    G = ox.graph_from_place('Greater London, UK', network_type='drive',
                            simplify=True)
    ox.save_graphml(G, GRAPH_CACHE)
    print(f"Downloaded: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

# Graph cleaning
n_before = G.number_of_nodes()
G = ox.truncate.largest_component(G, strongly=True)
print(f"Largest component: {G.number_of_nodes():,} nodes "
      f"({n_before - G.number_of_nodes():,} removed)")

self_loops = list(nx.selfloop_edges(G))
G.remove_edges_from(self_loops)
print(f"Self-loops removed: {len(self_loops):,}")

G_undirected = nx.Graph(G.to_undirected())
print(f"Final undirected graph: {G_undirected.number_of_nodes():,} nodes, "
      f"{G_undirected.number_of_edges():,} edges")


# =============================================================================
# 3.10b  Computing Centrality Measures
# =============================================================================

print("Converting NetworkX graph to iGraph...")
t0 = time.time()
nodes = list(G_undirected.nodes())
node_to_idx = {node: i for i, node in enumerate(nodes)}
edges = [(node_to_idx[u], node_to_idx[v])
         for u, v in G_undirected.edges()]
weights = [data.get('length', 1.0)
           for _, _, data in G_undirected.edges(data=True)]

g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
g_ig.es['length'] = weights
g_ig.vs['osmid'] = nodes
print(f"Converted in {time.time()-t0:.1f}s")

if CENTRALITY_CACHE.exists():
    print("Loading cached centrality measures...")
    centrality_df = pd.read_parquet(CENTRALITY_CACHE)
    print(f"Loaded: {len(centrality_df):,} nodes")
else:
    n_nodes = g_ig.vcount()
    t_total = time.time()
    print(f"Computing centrality for {n_nodes:,} nodes...")

    degree_vals = [d / (n_nodes - 1) for d in g_ig.degree()]
    print(f"  Degree done in {time.time()-t_total:.1f}s")

    t0 = time.time()
    bet_vals = g_ig.betweenness(
        directed=False, weights='length', normalized=True)
    print(f"  Betweenness done in {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    close_vals = g_ig.closeness(weights='length', normalized=True)
    print(f"  Closeness done in {(time.time()-t0)/60:.1f} min")

    centrality_df = pd.DataFrame({
        'node_id':               g_ig.vs['osmid'],
        'degree_centrality':     degree_vals,
        'closeness_centrality':  close_vals,
        'betweenness_centrality': bet_vals,
    })
    centrality_df.to_parquet(CENTRALITY_CACHE, index=False)
    print(f"Done in {(time.time()-t_total)/60:.1f} min — cached")


# =============================================================================
# 3.10c  Mapping Accidents to Graph Network
# =============================================================================
print("\nSnapping accidents to nearest network nodes...")
nearest_nodes = ox.nearest_nodes(
    G, X=gdf['longitude'].values, Y=gdf['latitude'].values
)
gdf['nearest_node'] = nearest_nodes

centrality_lookup = centrality_df.set_index('node_id')
gdf['degree_centrality'] = gdf['nearest_node'].map(
    centrality_lookup['degree_centrality'])
gdf['closeness_centrality'] = gdf['nearest_node'].map(
    centrality_lookup['closeness_centrality'])
gdf['betweenness_centrality'] = gdf['nearest_node'].map(
    centrality_lookup['betweenness_centrality'])

for col in ['degree_centrality', 'closeness_centrality', 'betweenness_centrality']:
    n = gdf[col].notna().sum()
    print(f"  {col}: {n:,} assigned ({n/len(gdf):.1%})")


# =============================================================================
# 3.11  Load Traffic Count Points & AADF
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 11: AADF traffic counts")
print("=" * 70)


TRAFFIC_TABLES = {
    "count_points": {
        "url":     "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/count_points.zip",
        "zip":     RAW_DIR / "count_points.zip",
        "csv":     RAW_DIR / "count_points.csv",
        "parquet": CACHE_DIR / "count_points.parquet",
    },
    "aadf": {
        "url":     "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip",
        "zip":     RAW_DIR / "dft_traffic_counts_aadf.zip",
        "csv":     RAW_DIR / "dft_traffic_counts_aadf.csv",
        "parquet": CACHE_DIR / "dft_traffic_counts_aadf.parquet",
    },
}


def load_traffic_table(name, cfg):
    if cfg["parquet"].exists():
        df = pd.read_parquet(cfg["parquet"])
        print(f"[{name}] Loaded from cache: {len(df):,} rows")
        return df
    if not cfg["csv"].exists():
        if not cfg["zip"].exists():
            print(f"[{name}] Downloading...")
            urllib.request.urlretrieve(cfg["url"], cfg["zip"])
        with zipfile.ZipFile(cfg["zip"], "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            z.extract(csv_files[0], RAW_DIR)
            extracted = RAW_DIR / csv_files[0]
            if extracted != cfg["csv"]:
                extracted.rename(cfg["csv"])
    df = pd.read_csv(cfg["csv"], low_memory=False)
    df.to_parquet(cfg["parquet"], index=False)
    print(f"[{name}] Loaded and cached: {len(df):,} rows")
    return df


cp_raw = load_traffic_table("count_points", TRAFFIC_TABLES["count_points"])
aadf_raw = load_traffic_table("aadf",         TRAFFIC_TABLES["aadf"])

cp_gdf = gpd.GeoDataFrame(
    cp_raw.dropna(subset=["longitude", "latitude"]),
    geometry=gpd.points_from_xy(
        cp_raw.dropna(subset=["longitude", "latitude"])["longitude"],
        cp_raw.dropna(subset=["longitude", "latitude"])["latitude"]
    ),
    crs="EPSG:4326"
)
cp_london = cp_gdf[cp_gdf.geometry.within(gla_boundary)].copy()
print(f"London count points: {len(cp_london):,}")

STUDY_YEARS = [2020, 2021, 2022, 2023, 2024]
london_cp_ids = set(cp_london["count_point_id"])

aadf_london = aadf_raw[
    aadf_raw["count_point_id"].isin(london_cp_ids) &
    aadf_raw["year"].isin(STUDY_YEARS)
].copy()
aadf_london.columns = aadf_london.columns.str.lower()

AADF_COLS = ["all_motor_vehicles", "all_hgvs", "pedal_cycles",
             "two_wheeled_motor_vehicles", "buses_and_coaches"]

aadf_site_mean = (
    aadf_london.groupby("count_point_id")[AADF_COLS]
    .mean().reset_index()
    .rename(columns={c: f"aadf_{c}" for c in AADF_COLS})
)

cp_meta = cp_london[["count_point_id", "road_category", "road_type",
                     "latitude", "longitude", "link_length_km"]].copy()
aadf_site_mean = aadf_site_mean.merge(cp_meta, on="count_point_id", how="left")

aadf_quality = (
    aadf_london.groupby("count_point_id")["estimation_method"]
    .apply(lambda x: "Estimated" if "Estimated" in x.values else "Counted")
    .reset_index()
    .rename(columns={"estimation_method": "aadf_estimation_method"})
)
aadf_site_mean = aadf_site_mean.merge(
    aadf_quality, on="count_point_id", how="left")
print(f"AADF site means: {len(aadf_site_mean):,} sites")


# =============================================================================
# 3.11b  AADF Spatial Join to Accidents
# =============================================================================

MAX_DIST_M = 1000


def haversine_metres(lat1, lon1, lat2, lon2):
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


site_coords_rad = np.radians(aadf_site_mean[["latitude", "longitude"]].values)
accident_coords_rad = np.radians(gdf[["latitude", "longitude"]].values)
tree = cKDTree(site_coords_rad)
_, idxs = tree.query(accident_coords_rad, k=1)

matched_lats = aadf_site_mean["latitude"].values[idxs]
matched_lons = aadf_site_mean["longitude"].values[idxs]
dist_m = haversine_metres(gdf["latitude"].values, gdf["longitude"].values,
                          matched_lats, matched_lons)

gdf["cp_distance_m"] = dist_m
gdf["cp_matched"] = dist_m <= MAX_DIST_M
gdf["nearest_cp_idx"] = idxs

aadf_site_reset = aadf_site_mean.reset_index(drop=True)
FEATURE_COLS_AADF = ["aadf_all_motor_vehicles", "aadf_all_hgvs",
                     "aadf_pedal_cycles", "aadf_two_wheeled_motor_vehicles",
                     "aadf_buses_and_coaches"]

for col in FEATURE_COLS_AADF:
    gdf[col] = np.nan
    gdf.loc[gdf["cp_matched"], col] = (
        aadf_site_reset[col].values[idxs[gdf["cp_matched"]]]
    )

gdf["aadf_log"] = np.log1p(gdf["aadf_all_motor_vehicles"])
zero_mask = gdf["cp_matched"] & (gdf["aadf_all_motor_vehicles"] == 0)
if zero_mask.sum() > 0:
    gdf.loc[zero_mask, FEATURE_COLS_AADF] = np.nan
    gdf["aadf_log"] = np.log1p(gdf["aadf_all_motor_vehicles"])

n_matched = gdf["cp_matched"].sum()
print(f"Matched (<=1000m): {n_matched:,} ({n_matched/len(gdf):.1%})")


# =============================================================================
# 3.11c  AADF Exposure Normalisation
# =============================================================================
STUDY_YEARS_N = 5

if "link_length_km" not in aadf_site_mean.columns:
    aadf_site_mean = aadf_site_mean.merge(
        cp_london[["count_point_id", "link_length_km"]],
        on="count_point_id", how="left"
    )
    aadf_site_reset = aadf_site_mean.reset_index(drop=True)

matched_gdf = gdf[gdf["cp_matched"] &
                  gdf["aadf_all_motor_vehicles"].notna()].copy()

acc_per_site = (
    matched_gdf.groupby("nearest_cp_idx")
    .agg(total_accidents=("collision_index", "count"),
         severe_accidents=("severity_binary", "sum"))
    .reset_index()
)

acc_per_site["aadf"] = (
    aadf_site_reset["aadf_all_motor_vehicles"].values[
        acc_per_site["nearest_cp_idx"]])
acc_per_site["link_length_km"] = (
    aadf_site_reset["link_length_km"].values[
        acc_per_site["nearest_cp_idx"]])

acc_per_site.loc[acc_per_site["link_length_km"]
                 == 0, "link_length_km"] = np.nan
acc_per_site.loc[acc_per_site["aadf"] == 0, "aadf"] = np.nan

has_length = acc_per_site["link_length_km"].notna(
) & acc_per_site["aadf"].notna()
has_aadf = ~acc_per_site["link_length_km"].notna(
) & acc_per_site["aadf"].notna()

acc_per_site.loc[has_length, "accident_rate"] = (
    acc_per_site.loc[has_length, "total_accidents"] * 1e6 /
    (acc_per_site.loc[has_length, "aadf"] *
     acc_per_site.loc[has_length, "link_length_km"] * 365 * STUDY_YEARS_N)
)
acc_per_site.loc[has_aadf, "accident_rate"] = (
    acc_per_site.loc[has_aadf, "total_accidents"] * 1e6 /
    (acc_per_site.loc[has_aadf, "aadf"] * 365 * STUDY_YEARS_N)
)

rate_lu = acc_per_site.set_index("nearest_cp_idx")["accident_rate"]
gdf["accident_rate"] = gdf["nearest_cp_idx"].map(rate_lu)
p99 = gdf["accident_rate"].quantile(0.99)
gdf["accident_rate_clipped"] = gdf["accident_rate"].clip(upper=p99)

r_freq = gdf["accident_rate_clipped"].corr(gdf["aadf_log"])
r_sev = gdf["severity_binary"].corr(gdf["aadf_log"])
print(f"r(aadf_log, accident_rate) = {r_freq:+.3f}")
print(f"r(aadf_log, severity)      = {r_sev:+.4f}")


# =============================================================================
# 3.12  Load LAEI 2022 Average Road Speeds
# =============================================================================
print("\n" + "=" * 70)
print("  STAGE 12: LAEI road speeds")
print("=" * 70)


MAJOR_PARQUET = CACHE_DIR / "laei2022_major_roads_speeds.parquet"

if MAJOR_PARQUET.exists():
    major_gla = gpd.read_parquet(MAJOR_PARQUET)
    print(f"[major roads] Loaded from cache: {len(major_gla):,} links")
else:
    emissions_gpkg = str(
        RAW_DIR / "LAEI2022-nox-pm-co2-major-roads-link-emissions.gpkg")
    major_geom = gpd.read_file(
        emissions_gpkg, layer="nox-major-roads-link-emissions", columns=["TOID"],
    )
    if major_geom.crs is None or major_geom.crs.to_epsg() != 27700:
        major_geom = major_geom.set_crs(epsg=27700)

    major_xlsx = str(RAW_DIR / "LAEI2022-major-roads-flows-and-speeds.xlsx")
    major_speeds = pd.read_excel(major_xlsx)

    MAJOR_COLS = [
        "TOID", "year", "laei.zone", "borough", "length.m",
        "road.classification", "speed.other.vehicles.kph", "speed.buses.kph",
        "AADF.motorcycles", "AADF.taxis", "AADF.cars", "AADF.phvs",
        "AADF.lgvs", "AADF.hgvs.rigid", "AADF.hgvs.articulated",
        "AADF.tfl.buses", "AADF.non.tfl.buses", "AADF.coaches",
    ]
    major_speeds = major_speeds[MAJOR_COLS]
    major_roads = major_geom.merge(major_speeds, on="TOID", how="inner")
    major_gla = major_roads[major_roads["laei.zone"] != "Non-GLA"].copy()

    aadf_cols = [c for c in major_gla.columns if c.startswith("AADF.")]
    major_gla["AADF.total"] = major_gla[aadf_cols].sum(axis=1)
    major_gla.to_parquet(MAJOR_PARQUET, index=False)
    print(f"[major roads] Built and cached: {len(major_gla):,} links")

print(f"  Speed (kph): mean={major_gla['speed.other.vehicles.kph'].mean():.1f}, "
      f"median={major_gla['speed.other.vehicles.kph'].median():.1f}")

# Minor roads speed grid
MINOR_PARQUET = CACHE_DIR / "laei2022_minor_roads_speed_grid.parquet"

if MINOR_PARQUET.exists():
    minor_grid = gpd.read_parquet(MINOR_PARQUET)
    print(f"[minor roads] Loaded from cache: {len(minor_grid):,} grid cells")
else:
    grid_shp = str(RAW_DIR / "New folder" / "LAEI2019_Grid.tab")
    grid_geom = gpd.read_file(grid_shp)
    if grid_geom.crs is None or grid_geom.crs.to_epsg() != 27700:
        grid_geom = grid_geom.set_crs(epsg=27700)

    minor_xlsx = str(RAW_DIR / "LAEI2022-minor-roads-flows-and-speeds.xlsx")
    minor_speeds = pd.read_excel(minor_xlsx)
    MINOR_COLS = [
        "grid.id.unique", "year", "grid.id.1km2", "easting", "northing",
        "area.km2", "area.m2", "laei.zone", "borough", "speed.kph",
    ]
    minor_speeds = minor_speeds[MINOR_COLS]
    minor_gla_speeds = minor_speeds[minor_speeds["laei.zone"]
                                    != "Non-GLA"].copy()
    minor_speed_dedup = minor_gla_speeds[["grid.id.1km2", "speed.kph", "borough", "laei.zone"]]\
        .drop_duplicates(subset=["grid.id.1km2"])

    grid_id_col = None
    for candidate in [c for c in grid_geom.columns if "1km2" in c.lower() or "grid" in c.lower()]:
        overlap = set(grid_geom[candidate]).intersection(
            set(minor_speed_dedup["grid.id.1km2"]))
        if len(overlap) > 100:
            grid_id_col = candidate
            break
    if grid_id_col is None:
        for col in grid_geom.columns:
            if col == "geometry":
                continue
            try:
                if len(set(grid_geom[col].dropna()).intersection(
                        set(minor_speed_dedup["grid.id.1km2"]))) > 100:
                    grid_id_col = col
                    break
            except TypeError:
                continue
    if grid_id_col is None:
        raise ValueError("Could not match grid ID between shapefile and Excel")

    minor_grid = grid_geom.merge(
        minor_speed_dedup, left_on=grid_id_col, right_on="grid.id.1km2", how="inner"
    )
    minor_grid.to_parquet(MINOR_PARQUET, index=False)
    print(f"[minor roads] Built and cached: {len(minor_grid):,} cells")

# Assign speed to each collision
MAJOR_ROAD_MATCH_TOLERANCE_M = 30
print(f"\nAssigning speeds to {len(gdf):,} collisions...")

gdf_bng = gdf.to_crs(epsg=27700) if gdf.crs.to_epsg() != 27700 else gdf.copy()

major_for_join = major_gla[["TOID", "speed.other.vehicles.kph", "speed.buses.kph",
                            "road.classification", "AADF.total", "geometry"]].copy()

matched = gpd.sjoin_nearest(
    gdf_bng[["geometry"]].copy().reset_index(),
    major_for_join, how="left",
    max_distance=MAJOR_ROAD_MATCH_TOLERANCE_M,
    distance_col="dist_to_major_road_m",
)
matched = matched.sort_values("AADF.total", ascending=False)\
    .drop_duplicates(subset=["index"], keep="first")\
    .set_index("index")

n_major = matched["TOID"].notna().sum()
print(f"  Major road matches (<=30m): {n_major:,} ({n_major/len(gdf):.1%})")

unmatched_mask = matched["TOID"].isna()
n_minor = 0
if unmatched_mask.sum() > 0:
    unmatched_pts = gdf_bng.loc[matched.loc[unmatched_mask].index, [
        "geometry"]].copy()
    minor_matched = gpd.sjoin(
        unmatched_pts, minor_grid[["speed.kph", "geometry"]], how="left", predicate="within"
    )
    minor_matched = minor_matched[~minor_matched.index.duplicated(
        keep="first")]
    n_minor = minor_matched["speed.kph"].notna().sum()
    print(f"  Minor grid matches: {n_minor:,}")

gdf["avg_speed_kph"] = matched["speed.other.vehicles.kph"]
gdf["speed_source"] = None
gdf.loc[matched["TOID"].notna(), "speed_source"] = "major_road_link"
gdf["dist_to_major_road_m"] = matched["dist_to_major_road_m"]
gdf["matched_road_class"] = matched["road.classification"]
gdf["matched_road_aadf"] = matched["AADF.total"]

if unmatched_mask.sum() > 0:
    still_missing = gdf["avg_speed_kph"].isna()
    gdf.loc[still_missing, "avg_speed_kph"] = minor_matched["speed.kph"]
    gdf.loc[still_missing & gdf["avg_speed_kph"].notna(
    ), "speed_source"] = "minor_road_grid"

n_with = gdf["avg_speed_kph"].notna().sum()
print(f"Speed assignment: {n_with:,}/{len(gdf):,} ({n_with/len(gdf):.1%})")

# =============================================================================
# OUTPUT — Save prepared data
# =============================================================================
print("\n" + "=" * 70)
print("  SAVING OUTPUTS")
print("=" * 70)

GDF_OUTPUT = CACHE_DIR / 'gdf_prepared.geoparquet'
CAS_OUTPUT = CACHE_DIR / 'casualties_ldn.parquet'

# Save gdf as GeoParquet
gdf.to_parquet(GDF_OUTPUT, index=False)
print(f"Saved: {GDF_OUTPUT}  ({GDF_OUTPUT.stat().st_size / 1e6:.1f} MB, "
      f"{len(gdf):,} rows x {gdf.shape[1]} cols)")

# Save casualties_ldn for EDA (raw casualty-level data)
casualties_ldn.to_parquet(CAS_OUTPUT, index=False)
print(f"Saved: {CAS_OUTPUT}  ({CAS_OUTPUT.stat().st_size / 1e6:.1f} MB, "
      f"{len(casualties_ldn):,} rows)")

print(f"\nFinal gdf columns ({gdf.shape[1]}):")
print(gdf.columns.to_list())
print("\nData preparation complete.")
