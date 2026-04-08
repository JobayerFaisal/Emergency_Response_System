"""
Satellite Imagery Service - GEE Sentinel-1 SAR Flood Detection
================================================================
Wraps Google Earth Engine SAR data retrieval + CNN flood detection
into an async-compatible service for the DataCollectionOrchestrator.

Uses the same GEE + change-detection approach as test_flood_detection.py
but packaged as a reusable collector that produces SatelliteData objects.

Author: Mahmudul / Environmental Intelligence Team
Version: 2.0.0
"""

import asyncio
import logging
import os
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

from backend.app.config.scenario import is_replay, get_scenario_date

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class FloodDetectionResult:
    """Result from satellite-based flood detection for a single zone"""
    zone_id: str
    zone_name: str
    bounds: List[float]  # [lon_min, lat_min, lon_max, lat_max]
    timestamp: str
    reference_date: Optional[str] = None
    current_date: Optional[str] = None
    flood_detected: bool = False
    flood_percentage: float = 0.0
    permanent_water_pct: float = 0.0
    current_water_pct: float = 0.0
    risk_level: str = "MINIMAL"  # MINIMAL, LOW, MEDIUM, HIGH, CRITICAL
    status: str = "NO SIGNIFICANT FLOODING"
    confidence: float = 0.0
    flood_area_km2: float = 0.0
    raw_predictions: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SatelliteData:
    """
    Aggregated satellite data for a sentinel zone.
    This is what gets merged with WeatherData in the orchestrator.
    """
    zone_id: str
    timestamp: datetime
    flood_detection: Optional[FloodDetectionResult] = None
    sar_available: bool = False
    source: str = "sentinel_1_gee"
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/Redis storage"""

        def _make_serializable(obj: Any) -> Any:
            """FIX: Recursively convert types that json.dumps() cannot handle.
            - numpy bool_ / bool → Python int (0 or 1)
            - numpy float32/float64 → Python float
            - numpy int types → Python int
            - dict/list are traversed recursively
            """
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            # numpy scalar types
            try:
                import numpy as np
                if isinstance(obj, (np.bool_,)):
                    return int(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            # Plain Python bool must come AFTER numpy check (bool is subclass of int)
            if isinstance(obj, bool):
                return int(obj)
            return obj

        result = {
            "zone_id": self.zone_id,
            "timestamp": self.timestamp.isoformat(),
            "sar_available": int(self.sar_available),   # bool → int
            "source": self.source,
            "processing_time_seconds": self.processing_time_seconds,
        }
        if self.flood_detection:
            result["flood_detection"] = _make_serializable(asdict(self.flood_detection))
        return result


# =====================================================================
# GEE SAR FUNCTIONS (extracted from test_flood_detection.py)
# =====================================================================

def _init_earth_engine(project_id: Optional[str] = None):
    """Initialize Earth Engine (lazy, thread-safe via GEE's own lock)"""
    import ee
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.info("Google Earth Engine initialized")
    except Exception as e:
        logger.error(f"GEE initialization failed: {e}")
        raise


def _download_sar_image(
    bounds: List[float],
    start_date: str,
    end_date: str
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Download Sentinel-1 SAR image from GEE.

    Args:
        bounds: [lon_min, lat_min, lon_max, lat_max]
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        Tuple of (sar_array[H,W,2], acquisition_date) or (None, None)
    """
    import ee

    geometry = ee.Geometry.Rectangle(bounds)

    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
    )

    count = collection.size().getInfo()
    if count == 0:
        logger.warning(f"No SAR images for bounds={bounds}, {start_date} to {end_date}")
        return None, None

    image = collection.median()

    first_img = collection.sort('system:time_start', False).first()
    timestamp = first_img.get('system:time_start').getInfo()
    acq_date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

    try:
        sample = image.sampleRectangle(region=geometry, defaultValue=-999).getInfo()
        vv = np.array(sample['properties']['VV'])
        vh = np.array(sample['properties']['VH'])
        vv[vv == -999] = np.nan
        vh[vh == -999] = np.nan
        return np.stack([vv, vh], axis=-1), acq_date
    except Exception as e:
        logger.error(f"SAR download error: {str(e)[:100]}")
        return None, None


def _preprocess_image(sar_image: np.ndarray, target_size: int = 64) -> Optional[np.ndarray]:
    """Preprocess SAR image for CNN model input"""
    from scipy.ndimage import zoom

    if sar_image is None:
        return None

    sar_image = np.nan_to_num(sar_image, nan=-15.0)

    h, w = sar_image.shape[:2]
    resized = zoom(sar_image, (target_size / h, target_size / w, 1), order=1)

    normalized = (resized + 35) / 35
    normalized = np.clip(np.asarray(normalized), 0, 1)

    return normalized.astype(np.float32)


def _detect_flood_change(
    model,
    reference_img: np.ndarray,
    current_img: np.ndarray
) -> Dict[str, np.ndarray]:
    """Detect floods using change detection between reference and current"""
    ref_pred = model.predict(np.expand_dims(reference_img, 0), verbose=0)[0, :, :, 0]
    cur_pred = model.predict(np.expand_dims(current_img, 0), verbose=0)[0, :, :, 0]

    ref_water = (ref_pred > 0.5).astype(np.float32)
    cur_water = (cur_pred > 0.5).astype(np.float32)

    flood_mask = np.maximum(0, cur_water - ref_water)
    vv_change = current_img[:, :, 0] - reference_img[:, :, 0]

    return {
        'ref_water': ref_pred,
        'cur_water': cur_pred,
        'flood_mask': flood_mask,
        'vv_change': vv_change,
    }


def _analyze_flood(detection: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze flood detection results and classify risk"""
    flood = detection['flood_mask']
    total = flood.size
    flooded = (flood > 0.5).sum()
    pct = (flooded / total) * 100

    perm_water = (detection['ref_water'] > 0.5).sum() / total * 100
    cur_water = (detection['cur_water'] > 0.5).sum() / total * 100

    if pct > 25:
        status, risk = "SEVERE FLOODING", "CRITICAL"
    elif pct > 15:
        status, risk = "SIGNIFICANT FLOODING", "HIGH"
    elif pct > 8:
        status, risk = "MODERATE FLOODING", "MEDIUM"
    elif pct > 3:
        status, risk = "MINOR FLOODING", "LOW"
    else:
        status, risk = "NO SIGNIFICANT FLOODING", "MINIMAL"

    return {
        'flood_pct': pct,
        'perm_water_pct': perm_water,
        'cur_water_pct': cur_water,
        'status': status,
        'risk': risk,
    }


# =====================================================================
# SATELLITE DATA COLLECTOR
# =====================================================================

class SatelliteDataCollector:
    """
    Async-compatible satellite data collector using GEE Sentinel-1 SAR.

    Integrates with the  alongside
    WeatherAPICollector and SocialMediaCollector.
    """

    def __init__(
        self,
        gee_project_id: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_client=None,
        cache_ttl: int = 3600,  # 1 hour (satellite data changes slowly)
    ):
        self.gee_project_id = gee_project_id or os.getenv(
            'GEE_PROJECT_ID', 'geecap'
        )
        self.model_path = model_path or os.getenv(
            'FLOOD_MODEL_PATH',
            str(Path(__file__).parent.parent / 'models' / 'flood_fast_best.keras')
        )
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl

        self._ee_initialized = False
        self._model = None
        self._model_loaded = False

        # Reference period (dry season for change detection baseline)
        self.ref_start = os.getenv('SAR_REF_START', '2024-01-01')
        self.ref_end = os.getenv('SAR_REF_END', '2024-03-31')

        logger.info("SatelliteDataCollector initialized")

    def _ensure_ee(self):
        """Lazy-init Earth Engine on first use"""
        if not self._ee_initialized:
            _init_earth_engine(self.gee_project_id)
            self._ee_initialized = True

    def _ensure_model(self):
        """Lazy-load the CNN model on first use"""
        if not self._model_loaded:
            import tensorflow as tf
            from tensorflow import keras

            def dice_coef(y_true, y_pred):
                y_true_f = tf.keras.backend.flatten(y_true)
                y_pred_f = tf.keras.backend.flatten(y_pred)
                intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
                return (2.0 * intersection + 1) / (
                    tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1
                )

            def dice_loss(y_true, y_pred):
                return 1 - dice_coef(y_true, y_pred)

            if os.path.exists(self.model_path):
                self._model = keras.models.load_model(
                    self.model_path,
                    custom_objects={
                        'dice_coef': dice_coef,
                        'dice_loss': dice_loss,
                    },
                )
                logger.info(f"Flood model loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}")
                self._model = None

            self._model_loaded = True

    def _zone_to_bounds(self, zone) -> List[float]:
        """Convert SentinelZone to GEE bounds [lon_min, lat_min, lon_max, lat_max]"""
        lat = zone.center.latitude
        lon = zone.center.longitude
        r_km = zone.radius_km

        dlat = r_km / 111.0
        dlon = r_km / (111.0 * np.cos(np.radians(lat)))

        return [
            round(lon - dlon, 4),
            round(lat - dlat, 4),
            round(lon + dlon, 4),
            round(lat + dlat, 4),
        ]

    def _run_flood_detection_sync(
        self,
        zone_id: str,
        zone_name: str,
        bounds: List[float],
    ) -> FloodDetectionResult:
        """
        Synchronous flood detection for a single zone.
        Called via asyncio.to_thread() for non-blocking execution.
        """
        self._ensure_ee()
        self._ensure_model()

        if self._model is None:
            return FloodDetectionResult(
                zone_id=zone_id,
                zone_name=zone_name,
                bounds=bounds,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error="Flood detection model not available",
            )

        now = datetime.now(timezone.utc)
        cur_end = now.strftime('%Y-%m-%d')
        cur_start = (now - timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            logger.info(f"[{zone_name}] Downloading reference SAR...")
            ref_raw, ref_date = _download_sar_image(bounds, self.ref_start, self.ref_end)
            if ref_raw is None:
                return FloodDetectionResult(
                    zone_id=zone_id, zone_name=zone_name, bounds=bounds,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error="No reference SAR data",
                )

            logger.info(f"[{zone_name}] Downloading current SAR...")
            cur_raw, cur_date = _download_sar_image(bounds, cur_start, cur_end)
            if cur_raw is None:
                return FloodDetectionResult(
                    zone_id=zone_id, zone_name=zone_name, bounds=bounds,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error="No current SAR data",
                )

            ref_proc = _preprocess_image(ref_raw, 64)
            cur_proc = _preprocess_image(cur_raw, 64)

            if ref_proc is None or cur_proc is None:
                return FloodDetectionResult(
                    zone_id=zone_id, zone_name=zone_name, bounds=bounds,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error="SAR preprocessing failed",
                )

            logger.info(f"[{zone_name}] Running change detection...")
            detection = _detect_flood_change(self._model, ref_proc, cur_proc)
            analysis = _analyze_flood(detection)

            # Estimate flood area in km²
            cos_lat = np.cos(np.radians((bounds[1] + bounds[3]) / 2))
            zone_w_km = (bounds[2] - bounds[0]) * 111.0 * cos_lat
            zone_h_km = (bounds[3] - bounds[1]) * 111.0
            total_area = zone_w_km * zone_h_km
            flood_area = total_area * (analysis['flood_pct'] / 100.0)

            logger.info(
                f"[{zone_name}] Done: {analysis['risk']} risk, "
                f"{analysis['flood_pct']:.1f}% flood, {flood_area:.2f} km²"
            )

            return FloodDetectionResult(
                zone_id=zone_id,
                zone_name=zone_name,
                bounds=bounds,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reference_date=ref_date,
                current_date=cur_date,
                flood_detected=analysis['flood_pct'] > 3.0,
                flood_percentage=round(analysis['flood_pct'], 2),
                permanent_water_pct=round(analysis['perm_water_pct'], 2),
                current_water_pct=round(analysis['cur_water_pct'], 2),
                risk_level=analysis['risk'],
                status=analysis['status'],
                confidence=min(0.95, 0.6 + (analysis['flood_pct'] / 100.0)),
                flood_area_km2=round(flood_area, 3),
            )

        except Exception as e:
            logger.error(f"[{zone_name}] Detection failed: {e}", exc_info=True)
            return FloodDetectionResult(
                zone_id=zone_id, zone_name=zone_name, bounds=bounds,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )



    async def fetch_satellite_data(self, zone, zone_id: Optional[str] = None) -> SatelliteData:
        """
        Fetch satellite flood detection data for a single zone.
        Non-blocking — runs GEE+TF calls in a thread pool.

        In replay mode, bypasses live GEE/CNN calls and returns deterministic
        historical-style flood conditions for demo stability.
        """
        from backend.app.config.scenario import is_replay, get_scenario_date

        zid = zone_id or str(zone.id)
        cache_key = f"satellite:{zid}"

        # -----------------------------------------------------------------
        # REPLAY MODE
        # -----------------------------------------------------------------
        if is_replay():
            logger.warning(f"[REPLAY MODE] Using simulated satellite data for {zone.name}")

            # Tune per zone risk level for a believable scenario
            risk_level_value = getattr(zone.risk_level, "value", str(zone.risk_level)).lower()

            if risk_level_value == "critical":
                flood_pct = 38.0
                risk_level = "CRITICAL"
            elif risk_level_value == "high":
                flood_pct = 24.0
                risk_level = "HIGH"
            elif risk_level_value == "moderate":
                flood_pct = 12.0
                risk_level = "MEDIUM"
            else:
                flood_pct = 5.5
                risk_level = "LOW"

            bounds = self._zone_to_bounds(zone)

            return SatelliteData(
                zone_id=zid,
                timestamp=datetime.fromisoformat(get_scenario_date().replace("Z", "+00:00")),
                flood_detection=FloodDetectionResult(
                    zone_id=zid,
                    zone_name=zone.name,
                    bounds=bounds,
                    timestamp=get_scenario_date(),
                    reference_date="2022-05-25",
                    current_date="2022-06-17",
                    flood_detected=True,
                    flood_percentage=flood_pct,
                    permanent_water_pct=4.5,
                    current_water_pct=min(flood_pct + 6.0, 100.0),
                    risk_level=risk_level,
                    status="REPLAY FLOOD EVENT",
                    confidence=0.95,
                    flood_area_km2=round((zone.radius_km ** 2) * 3.1416 * (flood_pct / 100.0), 3),
                    raw_predictions={
                        "mode": "replay_2022_sylhet",
                        "note": "Simulated historical SAR flood result for demo",
                    },
                    error=None,
                ),
                sar_available=True,
                source="replay_2022",
                processing_time_seconds=0.01,
            )

        # -----------------------------------------------------------------
        # LIVE MODE
        # -----------------------------------------------------------------

        # Check cache
        if self.cache_client:
            try:
                cached = await self.cache_client.get(cache_key)
                if cached:
                    logger.debug(f"Satellite cache hit for {zone.name}")
                    data = json.loads(cached)
                    return SatelliteData(
                        zone_id=zid,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        flood_detection=(
                            FloodDetectionResult(**data['flood_detection'])
                            if data.get('flood_detection') else None
                        ),
                        sar_available=data.get('sar_available', False),
                        source=data.get('source', 'sentinel_1_gee'),
                        processing_time_seconds=data.get('processing_time_seconds', 0),
                    )
            except Exception as e:
                logger.error(f"Satellite cache error: {e}")

        # Run in thread pool
        start = datetime.now(timezone.utc)
        bounds = self._zone_to_bounds(zone)

        flood_result = await asyncio.to_thread(
            self._run_flood_detection_sync, zid, zone.name, bounds,
        )

        processing_time = (datetime.now(timezone.utc) - start).total_seconds()

        satellite_data = SatelliteData(
            zone_id=zid,
            timestamp=datetime.now(timezone.utc),
            flood_detection=flood_result,
            sar_available=flood_result.error is None,
            source="sentinel_1_gee",
            processing_time_seconds=round(processing_time, 2),
        )

        # Cache
        if self.cache_client:
            try:
                await self.cache_client.setex(
                    cache_key, self.cache_ttl,
                    json.dumps(satellite_data.to_dict()),
                )
            except Exception as e:
                logger.error(f"Satellite cache store error: {e}")

        return satellite_data


    async def fetch_multiple_zones(self, zones: list) -> Dict[str, SatelliteData]:
        """Fetch satellite data for multiple zones (sequential for GEE rate limits)"""
        results: Dict[str, SatelliteData] = {}

        for zone in zones:
            zid = str(zone.id)
            try:
                data = await self.fetch_satellite_data(zone, zid)
                results[zid] = data
            except Exception as e:
                logger.error(f"Satellite fetch error for {zone.name}: {e}")
                results[zid] = SatelliteData(
                    zone_id=zid, timestamp=datetime.now(timezone.utc), sar_available=False,
                )

        return results