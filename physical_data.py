

import os, json, time, logging, warnings, argparse
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import datetime, timezone, timedelta
from shapely.geometry import Point
from shapely.validation import make_valid
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Suppress harmless PROJ/GDAL warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore', message='.*PROJ.*')
warnings.filterwarnings('ignore', message='.*proj.db.*')
os.environ.setdefault('GTIFF_SRS_SOURCE', 'EPSG')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('fiona').setLevel(logging.ERROR)

_SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    # ── Ignition ──────────────────────────────────────────────────────────
    'ignition_geojson': _SCRIPT_DIR / 'Forest_ignition.geojson',

    # ── Time window ───────────────────────────────────────────────────────
    'dt_minutes':  30,     # timestep interval
    'total_days':   7,     # how many days to collect

    # ── Weather APIs ──────────────────────────────────────────────────────
    'weather_url':   'https://firemap.sdsc.edu/pylaski/stations/data',
    'openmeteo_url': 'https://archive-api.open-meteo.com/v1/archive',

    # ── Output ────────────────────────────────────────────────────────────
    'output_dir':    _SCRIPT_DIR / 'fire_factor_data',
    'request_delay': 1,   # seconds between API calls (be polite)

    # ── Landscape rasters ─────────────────────────────────────────────────
    'landscape_dir': _SCRIPT_DIR / 'Forest_LCP_Outputs',
    'landscape_crs': 'EPSG:5070',
    'landscape_layers': {
        'elevation_m':          ['elevation_5070.tif'],
        'slope_pct':            ['slope_5070.tif'],
        'aspect_deg':           ['aspect_5070.tif'],
        'fuel_model':           ['fbfm40_5070.tif'],
        'canopy_cover_pct':     ['canopy_cover_5070.tif'],
        'canopy_height_m':      ['canopy_height_5070.tif'],
        'canopy_base_m':        ['canopy_base_height_5070.tif'],
        'canopy_density':       ['canopy_bulk_density_5070.tif'],
        'fuel_depth_m':         ['Forest_depth.tif'],
        'fuel_moist_1h':        ['Forest_moist1.tif'],
        'fuel_moist_10h':       ['Forest_moist10.tif'],
        'fuel_moist_100h':      ['Forest_moist100.tif'],
        'fuel_load_1h_kg_m2':   ['Forest_rhof1.tif'],
        'fuel_load_10h_kg_m2':  ['Forest_rhof10.tif'],
        'fuel_load_100h_kg_m2': ['Forest_rhof100.tif'],
        'fuel_sav_m2_m3':       ['Forest_SAV.tif'],
    },
    'surface_fuel_dir': _SCRIPT_DIR / 'forest-surface-fuels-and-surface-data',

    # ── Building data ─────────────────────────────────────────────────────
    'building_dir':    _SCRIPT_DIR / 'forest-building-data',
    'building_files':  [
        'Forest_generated_buildings_fireprops_addAttr.geojson',
        'Forest_generated_buildings_fireprops.geojson',
    ],
    'building_radius_m': 2000,

    # ── Road network ──────────────────────────────────────────────────────
    'road_dir':   _SCRIPT_DIR / 'forest-roads',
    'road_files': ['Forest_roads.geojson'],
    'road_radius_m': 2000,
}


# ===========================================================================
# A. Ignition loader
# ===========================================================================
def load_ignition_geojson(geojson_path):
    """
    Parse ignition-point GeoJSON -> list of ignition dicts.
    Required: Point geometry [lon, lat]
    Optional properties: start/ignition_time/datetime/date, label/name
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Ignition GeoJSON not found: {path.resolve()}")

    with open(path, encoding='utf-8') as f:
        gj = json.load(f)

    features = gj.get('features', [])
    if not features:
        raise ValueError(f"No features in {path}")

    ignitions = []
    for idx, feat in enumerate(features, start=1):
        geom  = feat.get('geometry', {})
        props = feat.get('properties') or {}

        if geom.get('type') != 'Point':
            log.warning(f"  Feature {idx}: not a Point — skipping")
            continue
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            log.warning(f"  Feature {idx}: missing coordinates — skipping")
            continue

        lon, lat = float(coords[0]), float(coords[1])
        label = (props.get('label') or props.get('name')
                 or props.get('fire_name') or props.get('FireName')
                 or f'ignition_{idx:03d}')

        raw_time = (props.get('start') or props.get('ignition_time')
                    or props.get('datetime') or props.get('date'))
        ignition_time = None
        if raw_time:
            try:
                ignition_time = pd.Timestamp(raw_time).to_pydatetime()
                if ignition_time.tzinfo is None:
                    ignition_time = ignition_time.replace(tzinfo=timezone.utc)
                else:
                    ignition_time = ignition_time.astimezone(timezone.utc)
            except Exception as e:
                log.warning(f"  Feature {idx}: bad timestamp '{raw_time}' — {e}")

        if ignition_time is None:
            log.warning(f"  Feature {idx}: no ignition time — using current UTC")
            ignition_time = datetime.now(timezone.utc)

        ignitions.append({
            'label':         label,
            'lat':           lat,
            'lon':           lon,
            'ignition_time': ignition_time,
        })
        log.info(f"  [{idx}] '{label}'  lat={lat}  lon={lon}  "
                 f"ignition={ignition_time.isoformat()}")

    if not ignitions:
        raise ValueError("No valid Point features found.")
    return ignitions


# ===========================================================================
# B. Timestep generator
# ===========================================================================
def generate_timesteps(ignition_time, dt_minutes=30, total_days=7):
    """
    Return list of datetime objects from ignition_time to
    ignition_time + total_days at dt_minutes intervals (inclusive start).

    Returns tz-naive UTC datetimes (for pandas compatibility).
    """
    # Convert to UTC tz-naive for pandas
    t0 = pd.Timestamp(ignition_time).tz_convert('UTC').tz_localize(None)
    total_steps = int(total_days * 24 * 60 / dt_minutes)
    return [t0 + pd.Timedelta(minutes=dt_minutes * i)
            for i in range(total_steps + 1)]  # +1 to include the endpoint


# ===========================================================================
# C. Weather fetchers
# ===========================================================================
def _fetch_pylaski(lat, lon, start_dt, end_dt):
    """Single-hour fetch from pylaski nearest station."""
    # pylaski needs at least 1-hour window
    if (end_dt - start_dt).total_seconds() < 3600:
        end_dt = start_dt + timedelta(hours=1)
    params = {
        'selection':  'closestTo',
        'lat':        f'{lat:.6f}',
        'lon':        f'{lon:.6f}',
        'observable': 'wind_speed,wind_direction,air_temp,relative_humidity',
        'from':       start_dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'to':         end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    try:
        resp = requests.get(CONFIG['weather_url'], params=params, timeout=20)
        resp.raise_for_status()
        text = resp.text.strip()
        if text.startswith('wxData(') and text.endswith(')'): text = text[7:-1]
        data    = json.loads(text)
        station = data['features'][0]['properties']
        geom    = data['features'][0].get('geometry')

        def avg(lst):
            lst = [v for v in (lst or []) if v is not None]
            return round(float(np.mean(lst)), 4) if lst else None

        return {
            'wind_speed_mph': avg(station.get('wind_speed')),
            'wind_dir_deg':   avg(station.get('wind_direction')),
            'temp_f':         avg(station.get('air_temp')),
            'humidity_pct':   avg(station.get('relative_humidity')),
            'station_name':   station.get('STATION_NAME',
                                          station.get('stationName', '')),
            'weather_source': 'pylaski',
        }
    except Exception as e:
        log.debug(f'    pylaski: {e}')
        return None


def _fetch_openmeteo_day(lat, lon, date_str):
    """
    Fetch a full day of hourly data from Open-Meteo archive API.
    Returns dict keyed by hour (0-23) -> weather dict.
    Efficient: one call per day covers all 30-min steps in that day.
    """
    params = {
        'latitude':        f'{lat:.6f}',
        'longitude':       f'{lon:.6f}',
        'start_date':      date_str,
        'end_date':        date_str,
        'hourly':          ('wind_speed_10m,wind_direction_10m,'
                            'temperature_2m,relative_humidity_2m'),
        'wind_speed_unit': 'mph',
        'timezone':        'UTC',
    }
    try:
        resp   = requests.get(CONFIG['openmeteo_url'], params=params, timeout=20)
        resp.raise_for_status()
        hourly = resp.json().get('hourly', {})
        times  = hourly.get('time', [])
        by_hour = {}
        for i, t in enumerate(times):
            h = int(t[11:13])
            tmp = (hourly['temperature_2m'][i]
                   if hourly.get('temperature_2m') and
                      i < len(hourly['temperature_2m']) else None)
            by_hour[h] = {
                'wind_speed_mph': (round(float(hourly['wind_speed_10m'][i]), 4)
                                   if hourly.get('wind_speed_10m') and
                                      hourly['wind_speed_10m'][i] is not None
                                   else None),
                'wind_dir_deg':   (round(float(hourly['wind_direction_10m'][i]), 4)
                                   if hourly.get('wind_direction_10m') and
                                      hourly['wind_direction_10m'][i] is not None
                                   else None),
                'temp_f':         (round(float(tmp) * 9/5 + 32, 2)
                                   if tmp is not None else None),
                'humidity_pct':   (round(float(hourly['relative_humidity_2m'][i]), 4)
                                   if hourly.get('relative_humidity_2m') and
                                      hourly['relative_humidity_2m'][i] is not None
                                   else None),
                'station_name':   f'Open-Meteo ({lat:.3f},{lon:.3f})',
                'weather_source': 'open-meteo',
            }
        return by_hour
    except Exception as e:
        log.debug(f'    open-meteo day {date_str}: {e}')
        return {}


_MISSING_WX = {
    'wind_speed_mph': None, 'wind_dir_deg': None,
    'temp_f': None, 'humidity_pct': None,
    'station_name': '', 'weather_source': 'missing',
}


class WeatherCache:
    """
    Cache Open-Meteo responses by (lat, lon, date) to avoid one API call
    per 30-min step.  Falls back to pylaski for recent/live data.
    """
    def __init__(self, lat, lon):
        self.lat      = lat
        self.lon      = lon
        self._om_days = {}   # date_str -> {hour -> wx_dict}

    def get(self, dt: pd.Timestamp) -> dict:
        """Return weather for a specific timestep."""
        # Try pylaski first (nearest station, best for historical point data)
        start = dt.to_pydatetime().replace(tzinfo=timezone.utc)
        end   = start + timedelta(hours=1)
        wx = _fetch_pylaski(self.lat, self.lon, start, end)
        if wx:
            return wx

        # Fall back to Open-Meteo (cached per day)
        date_str = dt.strftime('%Y-%m-%d')
        if date_str not in self._om_days:
            log.info(f'      [open-meteo] fetching day {date_str} ...')
            self._om_days[date_str] = _fetch_openmeteo_day(
                self.lat, self.lon, date_str)
            time.sleep(CONFIG['request_delay'])

        by_hour = self._om_days.get(date_str, {})
        if by_hour:
            # Use the closest hour; for 30-min steps interpolate by picking
            # the floor hour (conservative, avoids look-ahead)
            h = dt.hour
            if h in by_hour:
                return by_hour[h]
            # nearest available hour
            nearest = min(by_hour.keys(), key=lambda x: abs(x - h))
            return by_hour[nearest]

        return _MISSING_WX.copy()


# ===========================================================================
# D. Landscape sampler  (unchanged from v13, terrain/fuel are static)
# ===========================================================================
class LandscapeSampler:
    def __init__(self, landscape_dir, layers, crs='EPSG:5070',
                 surface_fuel_dir=None):
        self.landscape_dir     = Path(landscape_dir)
        self.layers            = layers
        self.crs               = crs
        self._cache            = {}
        self._available        = {}
        self._transformer      = Transformer.from_crs('EPSG:4326', crs,
                                                       always_xy=True)
        self._loaded           = False
        self._surface_fuel_dir = (Path(surface_fuel_dir)
                                  if surface_fuel_dir else None)

    def _find_file(self, candidates):
        for name in candidates:
            p = self.landscape_dir / name
            if p.exists(): return p
            if self._surface_fuel_dir:
                p2 = self._surface_fuel_dir / name
                if p2.exists(): return p2
        return None

    def _load_tif(self, path):
        try:
            import rasterio
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(path) as src:
                    data      = src.read(1).astype(float)
                    nodata    = src.nodata
                    transform = src.transform
            if nodata is not None:
                data[data == nodata] = np.nan
            return data, transform, 'tif'
        except ImportError:
            log.warning('rasterio not installed — pip install rasterio')
            return None, None, None

    def _load_asc(self, path):
        header = {}
        with open(path) as f:
            for _ in range(6):
                parts = f.readline().split()
                header[parts[0].lower()] = float(parts[1])
        data   = np.loadtxt(path, skiprows=6, dtype=float)
        nodata = header.get('nodata_value', -9999)
        data[data == nodata] = np.nan
        return data, header, 'asc'

    def load_all(self):
        log.info(f'Loading landscape rasters from: {self.landscape_dir}')
        if self._surface_fuel_dir:
            log.info(f'  (surface fuel fallback: {self._surface_fuel_dir})')
        for feat, candidates in self.layers.items():
            path = self._find_file(candidates)
            if path is None:
                log.warning(f'  {feat}: not found  (tried: {candidates})')
                continue
            log.info(f'  {feat}: {path.name}')
            if path.suffix == '.tif':
                data, transform, fmt = self._load_tif(path)
                if data is not None:
                    self._cache[feat]     = (data, transform, fmt)
                    self._available[feat] = path
            else:
                data, header, fmt = self._load_asc(path)
                self._cache[feat]     = (data, header, fmt)
                self._available[feat] = path
        self._loaded = True
        log.info(f'  Loaded {len(self._cache)}/{len(self.layers)} layers')

    def sample(self, lat, lon):
        if not self._loaded: self.load_all()
        x, y   = self._transformer.transform(lon, lat)
        result = {}
        for feat, (data, th, fmt) in self._cache.items():
            try:
                nrows, ncols = data.shape
                if fmt == 'tif':
                    t     = th
                    col_f = (x - t.c) / t.a
                    row_f = (y - t.f) / t.e
                else:
                    xll   = th.get('xllcorner', th.get('xllcenter', 0))
                    yll   = th.get('yllcorner', th.get('yllcenter', 0))
                    cs    = th['cellsize']
                    col_f = (x - xll) / cs
                    row_f = nrows - (y - yll) / cs - 1
                if (col_f < -0.5 or col_f > ncols + 0.5 or
                        row_f < -0.5 or row_f > nrows + 0.5):
                    result[feat] = None
                    continue
                col = int(np.clip(round(col_f), 0, ncols - 1))
                row = int(np.clip(round(row_f), 0, nrows - 1))
                val = data[row, col]
                result[feat] = None if np.isnan(val) else round(float(val), 4)
            except Exception as e:
                log.debug(f'  sample {feat}: {e}')
                result[feat] = None
        for feat in self.layers:
            if feat not in result:
                result[feat] = None
        return result

    def summary(self):
        lines = [f'Landscape sampler: {len(self._available)}/{len(self.layers)} layers']
        for feat in self.layers:
            s = (f'OK  {self._available[feat].name}'
                 if feat in self._available else 'MISSING')
            lines.append(f'  {feat:<22} {s}')
        return '\n'.join(lines)


# ===========================================================================
# E. Building sampler  (unchanged from v13)
# ===========================================================================
class BuildingSampler:
    FIELD_ALIASES = {
        'building height [m]': ['building height [m]'],
        'building width [m]':  ['building width [m]'],
        'building length [m]': ['building length [m]'],
        'woodRoof':            ['woodRoof', 'woodroof'],
        'woodSiding':          ['woodSiding', 'woodsiding'],
        'combustableDeck':     ['combustableDeck', 'combustabledeck'],
        'eaves':               ['eaves'],
        'ventScreens_in':      ['ventScreens_in', 'ventscreens_in'],
        'usage':               ['usage'],
    }

    def __init__(self, building_dir, filenames, radius_m=2000):
        self.building_dir = Path(building_dir)
        self.filenames    = filenames
        self.radius_m     = radius_m
        self._gdf         = None
        self._col_map     = {}
        self._loaded      = False

    def _resolve_col(self, gdf, aliases):
        for a in aliases:
            if a in gdf.columns: return a
        return None

    def load(self):
        for fname in self.filenames:
            path = self.building_dir / fname
            if path.exists():
                log.info(f'Loading building data: {path.name}')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    gdf = gpd.read_file(path)
                if gdf.crs is None:
                    gdf = gdf.set_crs('EPSG:4326')
                gdf = gdf.to_crs('EPSG:5070')
                for feat, aliases in self.FIELD_ALIASES.items():
                    col = self._resolve_col(gdf, aliases)
                    if col:
                        self._col_map[feat] = col
                gdf          = gdf.reset_index(drop=True)
                self._sindex = gdf.sindex
                self._gdf    = gdf
                self._loaded = True
                log.info(f'  {len(gdf)} buildings  cols={list(self._col_map.keys())}')
                return
        log.warning(f'No building file found in {self.building_dir}')
        self._loaded = True

    def sample(self, lat, lon):
        empty = {
            'building_count': 0, 'building_density_per_km2': 0.0,
            'avg_building_height_m': None, 'avg_building_width_m': None,
            'avg_building_length_m': None, 'frac_wood_roof': None,
            'frac_wood_siding': None, 'frac_combustable_deck': None,
            'frac_closed_eaves': None, 'avg_vent_screens_in': None,
            'frac_residential': None, 'frac_industrial': None,
            'dominant_usage': None, 'building_radius_m': self.radius_m,
        }
        if not self._loaded: self.load()
        if self._gdf is None or self._gdf.empty: return empty
        t      = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
        px, py = t.transform(lon, lat)
        circle = Point(px, py).buffer(self.radius_m)
        cand   = list(self._sindex.intersection(circle.bounds))
        if not cand: return empty
        subset = self._gdf.iloc[cand][
            self._gdf.iloc[cand].intersects(circle)].copy()
        n = len(subset)
        if n == 0: return empty
        ca = (np.pi * self.radius_m ** 2) / 1e6

        def smean(s):
            v = pd.to_numeric(s, errors='coerce').dropna()
            return round(float(v.mean()), 4) if len(v) else None
        def sfrac(s, val=1):
            v = pd.to_numeric(s, errors='coerce').dropna()
            return round(float((v == val).sum() / len(v)), 4) if len(v) else None
        def fstr(s, m):
            v = s.dropna()
            return (round(float((v.str.lower() == m.lower()).sum() / len(v)), 4)
                    if len(v) else None)
        def mstr(s):
            v = s.dropna()
            return v.mode().iloc[0] if len(v) else None

        ec = self._col_map.get('eaves'); fc = None
        if ec:
            es  = subset[ec].astype(str).str.strip().str.lower()
            tot = len(es.dropna())
            fc  = round(float((es == 'closed').sum() / tot), 4) if tot else None
        uc = self._col_map.get('usage'); fr = fi = du = None
        if uc:
            us = subset[uc].astype(str).str.strip()
            fr = fstr(us, 'Residential')
            fi = fstr(us, 'Industrial')
            du = mstr(us)
        return {
            'building_count':           n,
            'building_density_per_km2': round(n / ca, 2),
            'avg_building_height_m':    smean(subset[self._col_map['building height [m]']]) if 'building height [m]' in self._col_map else None,
            'avg_building_width_m':     smean(subset[self._col_map['building width [m]']])  if 'building width [m]'  in self._col_map else None,
            'avg_building_length_m':    smean(subset[self._col_map['building length [m]']]) if 'building length [m]' in self._col_map else None,
            'frac_wood_roof':           sfrac(subset[self._col_map['woodRoof']])        if 'woodRoof'        in self._col_map else None,
            'frac_wood_siding':         sfrac(subset[self._col_map['woodSiding']])      if 'woodSiding'      in self._col_map else None,
            'frac_combustable_deck':    sfrac(subset[self._col_map['combustableDeck']]) if 'combustableDeck' in self._col_map else None,
            'frac_closed_eaves':        fc,
            'avg_vent_screens_in':      smean(subset[self._col_map['ventScreens_in']])  if 'ventScreens_in'  in self._col_map else None,
            'frac_residential':  fr, 'frac_industrial': fi,
            'dominant_usage':    du, 'building_radius_m': self.radius_m,
        }

    def summary(self):
        if self._gdf is None: return 'Building sampler: not loaded'
        return (f'Building sampler: {len(self._gdf)} buildings  '
                f'radius={self.radius_m}m  cols={list(self._col_map.keys())}')


# ===========================================================================
# F. Road sampler  (unchanged from v13)
# ===========================================================================
class RoadSampler:
    HIGH_CAP   = {'motorway', 'trunk', 'primary', 'secondary',
                  'motorway_link', 'trunk_link', 'primary_link', 'secondary_link'}
    PAVED_SURF = {'asphalt', 'concrete', 'paved', 'concrete:plates',
                  'concrete:lanes', 'asphalt;concrete'}

    def __init__(self, road_dir, filenames, radius_m=2000):
        self.road_dir  = Path(road_dir)
        self.filenames = filenames
        self.radius_m  = radius_m
        self._gdf      = None
        self._sindex   = None
        self._loaded   = False

    def _parse_other_tags(self, tag_str):
        result = {}
        if not tag_str or not isinstance(tag_str, str): return result
        for part in tag_str.split(','):
            if '=>' in part:
                k, v = part.strip().split('=>', 1)
                result[k.strip().strip('"')] = v.strip().strip('"')
        return result

    def load(self):
        for fname in self.filenames:
            path = self.road_dir / fname
            if path.exists():
                log.info(f'Loading road data: {path.name}')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    gdf = gpd.read_file(path)
                if gdf.crs is None: gdf = gdf.set_crs('EPSG:4326')
                gdf = gdf.to_crs('EPSG:5070')
                if 'other_tags' in gdf.columns:
                    parsed         = gdf['other_tags'].apply(self._parse_other_tags)
                    gdf['_surface'] = parsed.apply(lambda d: d.get('surface', ''))
                    gdf['_lanes']   = parsed.apply(
                        lambda d: float(d['lanes']) if 'lanes' in d else np.nan)
                else:
                    gdf['_surface'] = ''; gdf['_lanes'] = np.nan
                if 'width'   not in gdf.columns: gdf['width']   = np.nan
                if 'highway' not in gdf.columns: gdf['highway'] = ''
                gdf              = gdf.explode(index_parts=False).reset_index(drop=True)
                gdf['_length_m'] = gdf.geometry.length
                self._sindex     = gdf.sindex
                self._gdf        = gdf
                self._loaded     = True
                log.info(f'  {len(gdf)} road segments  '
                         f'total={gdf["_length_m"].sum()/1000:.1f} km')
                return
        log.warning(f'No road file found in {self.road_dir}')
        self._loaded = True

    def sample(self, lat, lon):
        empty = {
            'road_count': 0, 'total_road_length_m': 0.0,
            'road_density_m_per_km2': 0.0, 'nearest_road_m': None,
            'frac_high_capacity': 0.0, 'frac_residential_road': 0.0,
            'frac_paved': 0.0, 'avg_road_width_m': None,
            'avg_lanes': None, 'dominant_highway': None,
            'road_radius_m': self.radius_m,
        }
        if not self._loaded: self.load()
        if self._gdf is None or self._gdf.empty: return empty
        t      = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
        px, py = t.transform(lon, lat); pt = Point(px, py)
        circle = pt.buffer(self.radius_m)
        ca     = (np.pi * self.radius_m ** 2) / 1e6
        cand   = list(self._sindex.intersection(circle.bounds))
        if not cand: return empty
        subset = self._gdf.iloc[cand][
            self._gdf.iloc[cand].intersects(circle)].copy()
        n = len(subset)
        if n == 0: return empty
        subset['_cl'] = subset.geometry.intersection(circle).length
        total = subset['_cl'].sum()
        hw    = subset['highway'].fillna('').str.lower()
        su    = subset['_surface'].fillna('').str.lower()
        wi    = pd.to_numeric(subset['width'], errors='coerce').dropna()
        la    = subset['_lanes'].dropna()
        return {
            'road_count':             n,
            'total_road_length_m':    round(total, 1),
            'road_density_m_per_km2': round(total / ca, 1),
            'nearest_road_m':         round(float(subset.geometry.distance(pt).min()), 1),
            'frac_high_capacity':     round(float(hw.isin(self.HIGH_CAP).sum() / n), 4),
            'frac_residential_road':  round(float((hw == 'residential').sum() / n), 4),
            'frac_paved':             round(float(su.isin(self.PAVED_SURF).sum() / n), 4),
            'avg_road_width_m':       round(float(wi.mean()), 2) if len(wi) else None,
            'avg_lanes':              round(float(la.mean()), 2) if len(la) else None,
            'dominant_highway':       hw.value_counts().index[0] if len(hw) > 0 else None,
            'road_radius_m':          self.radius_m,
        }

    def summary(self):
        if self._gdf is None: return 'Road sampler: not loaded'
        return (f'Road sampler: {len(self._gdf)} segments  '
                f'radius={self.radius_m}m  '
                f'total={self._gdf["_length_m"].sum()/1000:.1f} km')


# ===========================================================================
# G. Main collection pipeline
# ===========================================================================
def collect_factors(config):
    """
    For each ignition point:
      1. Generate 30-min timesteps for 7 days
      2. Sample weather at each timestep  (cached per day via Open-Meteo)
      3. Sample terrain/fuel once        (static at ignition centroid)
      4. Sample buildings/roads once     (static at ignition centroid)
      5. Assemble flat row per timestep

    Returns list of row dicts.
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load static samplers once ────────────────────────────────────────
    lsampler = LandscapeSampler(
        landscape_dir    = config['landscape_dir'],
        layers           = config['landscape_layers'],
        crs              = config['landscape_crs'],
        surface_fuel_dir = config.get('surface_fuel_dir'),
    )
    lsampler.load_all()
    log.info('\n' + lsampler.summary())

    bsampler = BuildingSampler(
        config['building_dir'],
        config['building_files'],
        config['building_radius_m'],
    )
    bsampler.load()
    log.info('\n' + bsampler.summary())

    rsampler = RoadSampler(
        config['road_dir'],
        config['road_files'],
        config['road_radius_m'],
    )
    rsampler.load()
    log.info('\n' + rsampler.summary())

    all_rows = []

    for ign in config['ignitions']:
        label         = ign['label']
        lat           = ign['lat']
        lon           = ign['lon']
        ignition_time = ign['ignition_time']

        log.info(f'\n{"="*60}')
        log.info(f'Ignition : {label}  lat={lat}  lon={lon}')
        log.info(f'Start    : {ignition_time.isoformat()}')
        log.info(f'Steps    : every {config["dt_minutes"]} min  '
                 f'for {config["total_days"]} days  '
                 f'= {int(config["total_days"]*24*60/config["dt_minutes"])+1} rows')
        log.info(f'{"="*60}')

        # ── Static features (terrain, fuel, buildings, roads) ────────────
        log.info('  Sampling static features (terrain / fuel / buildings / roads) ...')
        lf = lsampler.sample(lat, lon)
        bf = bsampler.sample(lat, lon)
        rf = rsampler.sample(lat, lon)
        log.info(f'    terrain: elev={lf.get("elevation_m")}m  '
                 f'slope={lf.get("slope_pct")}%  '
                 f'fuel={lf.get("fuel_model")}  '
                 f'cover={lf.get("canopy_cover_pct")}%')
        log.info(f'    buildings: {bf["building_count"]}  '
                 f'roads: {rf["road_count"]}')

        # ── Weather cache for this ignition ──────────────────────────────
        wx_cache = WeatherCache(lat, lon)

        # ── Timestep loop ────────────────────────────────────────────────
        timesteps = generate_timesteps(
            ignition_time,
            dt_minutes  = config['dt_minutes'],
            total_days  = config['total_days'],
        )
        log.info(f'  Collecting weather for {len(timesteps)} timesteps ...')

        for step_idx, ts in enumerate(timesteps):
            elapsed_h = step_idx * config['dt_minutes'] / 60.0

            wx = wx_cache.get(ts)

            if step_idx % 48 == 0:   # log every 24 h
                log.info(f'    step {step_idx:4d}  {ts}  '
                         f'elapsed={elapsed_h:.1f}h  '
                         f'wx={wx["weather_source"]}  '
                         f'wind={wx["wind_speed_mph"]} mph')

            # ── Assemble flat row ────────────────────────────────────────
            row = {
                # ── Meta ──────────────────────────────────────────────
                'label':          label,
                'ignition_lat':   lat,
                'ignition_lon':   lon,
                'ignition_time':  ignition_time.isoformat(),
                'timestep_index': step_idx,
                'datetime':       str(ts),
                'elapsed_hours':  round(elapsed_h, 4),

                # ── Weather (dynamic per timestep) ────────────────────
                'wind_speed_mph': wx['wind_speed_mph'],
                'wind_dir_deg':   wx['wind_dir_deg'],
                'temp_f':         wx['temp_f'],
                'humidity_pct':   wx['humidity_pct'],
                'weather_source': wx['weather_source'],

                # ── Terrain (static) ──────────────────────────────────
                'elevation_m':  lf.get('elevation_m'),
                'slope_pct':    lf.get('slope_pct'),
                'aspect_deg':   lf.get('aspect_deg'),
                'fuel_model':   lf.get('fuel_model'),

                # ── Canopy (static) ───────────────────────────────────
                'canopy_cover_pct': lf.get('canopy_cover_pct'),
                'canopy_height_m':  lf.get('canopy_height_m'),
                'canopy_base_m':    lf.get('canopy_base_m'),
                'canopy_density':   lf.get('canopy_density'),

                # ── Surface fuel (static) ─────────────────────────────
                'fuel_depth_m':         lf.get('fuel_depth_m'),
                'fuel_moist_1h':        lf.get('fuel_moist_1h'),
                'fuel_moist_10h':       lf.get('fuel_moist_10h'),
                'fuel_moist_100h':      lf.get('fuel_moist_100h'),
                'fuel_load_1h_kg_m2':   lf.get('fuel_load_1h_kg_m2'),
                'fuel_load_10h_kg_m2':  lf.get('fuel_load_10h_kg_m2'),
                'fuel_load_100h_kg_m2': lf.get('fuel_load_100h_kg_m2'),
                'fuel_sav_m2_m3':       lf.get('fuel_sav_m2_m3'),

                # ── Buildings (static) ────────────────────────────────
                **{f'bldg_{k}' if k == 'building_radius_m' else k: v
                   for k, v in bf.items()},

                # ── Roads (static) ────────────────────────────────────
                **{f'road_{k}' if k == 'road_radius_m' else k: v
                   for k, v in rf.items()},
            }
            all_rows.append(row)

            # Polite delay only on actual API calls (pylaski per step)
            # Open-Meteo is cached per day so no per-step delay needed
            if wx['weather_source'] == 'pylaski':
                time.sleep(config['request_delay'])

        log.info(f'  Done: {len(timesteps)} rows collected for "{label}"')

    return all_rows


# ===========================================================================
# H. Save outputs
# ===========================================================================
def save_outputs(rows, config):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        log.warning('No rows to save.')
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = output_dir / 'fire_factors.csv'
    df.to_csv(csv_path, index=False)
    log.info(f'\nSaved CSV  : {csv_path}  ({len(df)} rows x {len(df.columns)} cols)')

    # ── JSON (with metadata) ──────────────────────────────────────────────
    wx_src = df['weather_source'].value_counts().to_dict() if 'weather_source' in df else {}
    meta = {
        'created_at':     datetime.now(timezone.utc).isoformat(),
        'total_rows':     len(df),
        'dt_minutes':     config['dt_minutes'],
        'total_days':     config['total_days'],
        'ignitions':      [{'label': i['label'],
                            'lat':   i['lat'],
                            'lon':   i['lon'],
                            'ignition_time': i['ignition_time'].isoformat()}
                           for i in config['ignitions']],
        'weather_sources': wx_src,
        'columns':        list(df.columns),
        'note': (
            'Static columns (terrain, fuel, canopy, buildings, roads) repeat '
            'each row — they are spatially constant at the ignition point. '
            'Weather columns vary per timestep. '
            'Feed each row as input to your physical spread model to predict '
            'the fire perimeter at that timestep.'
        ),
    }
    json_path = output_dir / 'fire_factors.json'
    # Store as records for easy loading
    out = {'metadata': meta, 'records': rows}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log.info(f'Saved JSON : {json_path}')

    # ── Coverage summary ──────────────────────────────────────────────────
    log.info('\nCoverage summary:')
    check_cols = ['wind_speed_mph', 'wind_dir_deg', 'temp_f', 'humidity_pct',
                  'elevation_m', 'slope_pct', 'fuel_model',
                  'canopy_cover_pct', 'building_count', 'road_count']
    for col in check_cols:
        if col in df.columns:
            n_ok = df[col].notna().sum()
            log.info(f'  {col:<28} {n_ok}/{len(df)} non-null')

    return df


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wildfire physical-model factor collector — ignition to 1 week, 30-min steps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use ignition GeoJSON (must have a "start" timestamp property)
  python fire_factor_collector.py --ignition Forest_ignition.geojson

  # Override ignition point directly
  python fire_factor_collector.py \\
      --ignition-lat 38.9014 --ignition-lon -120.0306 \\
      --ignition-time "2024-10-27T10:00:00-0700"

  # Change time resolution or window
  python fire_factor_collector.py --dt-minutes 60 --days 3

  # Custom output directory
  python fire_factor_collector.py --output-dir ./my_factors
        """)

    parser.add_argument('--ignition',      type=Path,  default=CONFIG['ignition_geojson'],
                        help='Ignition point GeoJSON  (default: %(default)s)')
    parser.add_argument('--ignition-lat',  type=float, default=None,
                        help='Override ignition latitude')
    parser.add_argument('--ignition-lon',  type=float, default=None,
                        help='Override ignition longitude')
    parser.add_argument('--ignition-time', type=str,   default=None,
                        help='Override ignition time  e.g. "2024-10-27T10:00:00-0700"')
    parser.add_argument('--dt-minutes',    type=int,   default=CONFIG['dt_minutes'],
                        help='Timestep interval in minutes  (default: %(default)s)')
    parser.add_argument('--days',          type=float, default=CONFIG['total_days'],
                        help='Collection window in days  (default: %(default)s)')
    parser.add_argument('--output-dir',    type=Path,  default=CONFIG['output_dir'],
                        help='Output directory  (default: %(default)s)')
    args = parser.parse_args()

    def _resolve(p):
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else _SCRIPT_DIR / p

    CONFIG['dt_minutes']  = args.dt_minutes
    CONFIG['total_days']  = args.days
    CONFIG['output_dir']  = _resolve(args.output_dir)

    # ── Resolve ignitions ─────────────────────────────────────────────────
    if args.ignition_lat is not None and args.ignition_lon is not None:
        # CLI lat/lon override
        raw_time = args.ignition_time or '2024-10-27T10:00:00-0700'
        ign_time = pd.Timestamp(raw_time).to_pydatetime()
        if ign_time.tzinfo is None:
            ign_time = ign_time.replace(tzinfo=timezone.utc)
        else:
            ign_time = ign_time.astimezone(timezone.utc)
        CONFIG['ignitions'] = [{
            'label':         'cli_ignition',
            'lat':           args.ignition_lat,
            'lon':           args.ignition_lon,
            'ignition_time': ign_time,
        }]
        log.info(f'CLI ignition: lat={args.ignition_lat} lon={args.ignition_lon} '
                 f'time={ign_time.isoformat()}')
    else:
        gjson = _resolve(args.ignition)
        log.info(f'Loading ignition GeoJSON: {gjson}')
        CONFIG['ignitions'] = load_ignition_geojson(gjson)
        log.info(f'Loaded {len(CONFIG["ignitions"])} ignition(s)')

    # ── Validate landscape directory ──────────────────────────────────────
    if not Path(CONFIG['landscape_dir']).exists():
        log.error(f'Landscape dir missing: {Path(CONFIG["landscape_dir"]).resolve()}')
        raise SystemExit(1)

    total_steps = int(CONFIG['total_days'] * 24 * 60 / CONFIG['dt_minutes']) + 1
    log.info(f'\nCollection plan:')
    log.info(f'  dt_minutes  : {CONFIG["dt_minutes"]} min')
    log.info(f'  total_days  : {CONFIG["total_days"]} days')
    log.info(f'  total_steps : {total_steps} per ignition')
    log.info(f'  ignitions   : {len(CONFIG["ignitions"])}')
    log.info(f'  total rows  : {total_steps * len(CONFIG["ignitions"])}')
    log.info(f'  output_dir  : {CONFIG["output_dir"]}')

    # ── Run ───────────────────────────────────────────────────────────────
    rows = collect_factors(CONFIG)
    df   = save_outputs(rows, CONFIG)

    log.info('\n' + '='*60)
    log.info('Collection complete')
    log.info(f'  Total rows : {len(df)}')
    log.info(f'  Columns    : {len(df.columns)}')
    log.info(f'  Output     : {CONFIG["output_dir"]}')
    log.info('='*60)