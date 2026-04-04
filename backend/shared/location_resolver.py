"""
shared/location_resolver.py

CORNER CASE HANDLER: Resolves vague area names (e.g. "Mirpur", "Sylhet")
to exact sub-zone coordinates.

Problem: Agent 2's NLP extracts "Mirpur, Dhaka" from a message, but
Mirpur spans 12+ sectors covering ~20 km². Sending a rescue team to
the center of Mirpur is useless.

Solution: 3-layer resolution pipeline
  Layer 1 — Keyword sub-zone extraction from raw message text
  Layer 2 — Landmark/street name lookup table
  Layer 3 — Fallback to area centroid with radius uncertainty flag
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResolvedLocation:
    display_name: str          # Human-readable, e.g. "Mirpur Section 10, Dhaka"
    latitude: float
    longitude: float
    confidence: float          # 0.0 – 1.0
    resolution_method: str     # "subzone_keyword" | "landmark" | "area_centroid"
    uncertainty_radius_m: int  # How uncertain is this coordinate?
    raw_input: str             # Original text for audit trail
    needs_followup: bool       # True = Agent 2 should ask for more detail


# ─────────────────────────────────────────────────────────────────────────────
# SUB-ZONE DATABASE
# Each entry: (display_name, lat, lon, uncertainty_radius_m)
# Sources: OpenStreetMap Bangladesh + BIWTA flood maps
# ─────────────────────────────────────────────────────────────────────────────

SUBZONE_DB: dict[str, list[tuple]] = {

    # ── MIRPUR (most problematic — 12 sectors) ────────────────────────────
    "mirpur": [
        ("Mirpur-1, Dhaka",         23.7935, 90.3540, 400),
        ("Mirpur-2, Dhaka",         23.8041, 90.3654, 400),
        ("Mirpur-6, Dhaka",         23.8170, 90.3660, 400),
        ("Mirpur-7, Dhaka",         23.8100, 90.3580, 400),
        ("Mirpur-10 Circle, Dhaka", 23.8058, 90.3689, 300),  # Major landmark
        ("Mirpur-11, Dhaka",        23.8169, 90.3709, 400),
        ("Mirpur-12, Dhaka",        23.8249, 90.3710, 400),
        ("Mirpur-13, Dhaka",        23.8347, 90.3742, 400),
        ("Pallabi, Mirpur, Dhaka",  23.8270, 90.3610, 500),
        ("Mirpur DOHS, Dhaka",      23.8221, 90.3791, 500),
        ("Shewrapara, Mirpur",      23.7924, 90.3595, 400),
        ("Kafrul, Mirpur, Dhaka",   23.7919, 90.3691, 500),
        ("Kazipara, Mirpur, Dhaka", 23.8000, 90.3620, 400),
    ],

    # ── DHAKA (other major zones) ─────────────────────────────────────────
    "mohakhali": [("Mohakhali, Dhaka", 23.7781, 90.4070, 300)],
    "gulshan":   [("Gulshan, Dhaka",   23.7806, 90.4152, 500)],
    "banani":    [("Banani, Dhaka",    23.7937, 90.4066, 300)],
    "uttara":    [("Uttara, Dhaka",    23.8759, 90.3795, 600)],
    "badda":     [("Badda, Dhaka",     23.7787, 90.4337, 400)],
    "rampura":   [("Rampura, Dhaka",   23.7563, 90.4271, 400)],
    "demra":     [("Demra, Dhaka",     23.7133, 90.4700, 500)],
    "jatrabari": [("Jatrabari, Dhaka", 23.7117, 90.4327, 400)],
    "post office": [("GPO, Dhaka",     23.7239, 90.4084, 200)],
    "dhanmondi": [("Dhanmondi, Dhaka", 23.7461, 90.3742, 500)],
    "mohammadpur": [("Mohammadpur, Dhaka", 23.7629, 90.3573, 500)],
    "lalbagh":   [("Lalbagh, Dhaka",   23.7187, 90.3927, 400)],
    "sadarghat": [("Sadarghat, Dhaka", 23.7104, 90.4074, 200)],
    "kamrangirchar": [("Kamrangirchar, Dhaka", 23.7049, 90.3814, 500)],
    "keraniganj": [("Keraniganj, Dhaka", 23.6922, 90.3795, 700)],
    "tongi":     [("Tongi, Gazipur",   23.8927, 90.3987, 600)],
    "kawran bazar": [("Kawran Bazar, Dhaka", 23.7514, 90.3930, 200)],
    "farmgate":  [("Farmgate, Dhaka",  23.7577, 90.3870, 200)],
    "tejgaon":   [("Tejgaon, Dhaka",   23.7682, 90.3992, 300)],
    "motijheel": [("Motijheel, Dhaka", 23.7328, 90.4187, 300)],

    # ── SYLHET ────────────────────────────────────────────────────────────
    "sylhet": [
        ("Sylhet City Centre",          24.8998, 91.8710, 600),
        ("Sylhet Station Road",         24.8975, 91.8720, 200),
        ("Sylhet Akhalia",              24.9144, 91.8357, 500),
        ("Sylhet Shahjalal University", 24.9187, 91.8348, 400),
        ("Sylhet Ambarkhana",           24.8960, 91.8784, 300),
        ("Sylhet Zindabazar",           24.8949, 91.8760, 200),
        ("Sylhet Upashahar",            24.9044, 91.8643, 400),
        ("Sylhet Tilagor",              24.8800, 91.8900, 400),
    ],
    "sunamganj": [("Sunamganj Town",    25.0715, 91.3953, 700)],
    "habiganj":  [("Habiganj Town",     24.3744, 91.4153, 700)],
    "moulvibazar": [("Moulvibazar Town", 24.4826, 91.7774, 700)],

    # ── CHITTAGONG ────────────────────────────────────────────────────────
    "chittagong": [
        ("Chittagong City Centre", 22.3384, 91.8317, 700),
        ("Agrabad, Chittagong",    22.3250, 91.8130, 300),
        ("Patenga, Chittagong",    22.2434, 91.7881, 400),
        ("Halishahar, Chittagong", 22.3581, 91.7856, 400),
    ],
    "cox's bazar": [("Cox's Bazar",      21.4272, 92.0058, 800)],

    # ── BARISHAL ──────────────────────────────────────────────────────────
    "barishal": [
        ("Barishal City Centre",   22.7010, 90.3535, 600),
        ("Barishal Nathullabad",   22.7176, 90.3756, 300),
    ],

    # ── SIRAJGANJ (key flood zone) ────────────────────────────────────────
    "sirajganj": [
        ("Sirajganj Town",         24.4534, 89.7007, 600),
        ("Sirajganj Sadar",        24.4600, 89.7100, 400),
        ("Sirajganj River Bank",   24.4490, 89.6950, 300),
        ("Enayetpur, Sirajganj",   24.3747, 89.6850, 500),
    ],

    # ── JAMALPUR / BOGRA (frequent flood zones) ───────────────────────────
    "jamalpur": [("Jamalpur Town",  24.9000, 89.9378, 700)],
    "bogra":    [("Bogra Town",     24.8520, 89.3692, 700)],
    "gaibandha": [("Gaibandha Town", 25.3286, 89.5284, 700)],
    "kurigram": [("Kurigram Town",  25.8074, 89.6357, 700)],

    # ── NARAYANGANJ ───────────────────────────────────────────────────────
    "narayanganj": [("Narayanganj City", 23.6238, 90.4986, 600)],
}


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK LOOKUP (roads, stations, schools, hospitals)
# For messages like "near Sylhet station" or "Mirpur 10 circle"
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_DB: dict[str, tuple] = {
    # Mirpur landmarks
    "mirpur 10":          ("Mirpur-10 Circle, Dhaka",         23.8058, 90.3689, 150),
    "mirpur 10 circle":   ("Mirpur-10 Circle, Dhaka",         23.8058, 90.3689, 100),
    "mirpur 1":           ("Mirpur-1, Dhaka",                 23.7935, 90.3540, 200),
    "mirpur 2":           ("Mirpur-2, Dhaka",                 23.8041, 90.3654, 200),
    "mirpur 6":           ("Mirpur-6, Dhaka",                 23.8170, 90.3660, 200),
    "mirpur 11":          ("Mirpur-11, Dhaka",                23.8169, 90.3709, 200),
    "mirpur 12":          ("Mirpur-12, Dhaka",                23.8249, 90.3710, 200),
    "mirpur 13":          ("Mirpur-13, Dhaka",                23.8347, 90.3742, 200),
    "pallabi":            ("Pallabi, Mirpur, Dhaka",          23.8270, 90.3610, 300),
    "shewrapara":         ("Shewrapara, Mirpur, Dhaka",       23.7924, 90.3595, 200),
    "kazipara":           ("Kazipara, Mirpur, Dhaka",         23.8000, 90.3620, 200),
    "kafrul":             ("Kafrul, Mirpur, Dhaka",           23.7919, 90.3691, 300),
    "mirpur dohs":        ("Mirpur DOHS, Dhaka",              23.8221, 90.3791, 300),
    "section 10":         ("Mirpur Section 10, Dhaka",        23.8058, 90.3689, 150),
    "section 11":         ("Mirpur Section 11, Dhaka",        23.8169, 90.3709, 150),
    "section 12":         ("Mirpur Section 12, Dhaka",        23.8249, 90.3710, 150),

    # Sylhet landmarks
    "sylhet station":     ("Sylhet Railway Station",          24.8975, 91.8720, 100),
    "osmani":             ("Sylhet MAG Osmani Hospital",      24.8998, 91.8710, 150),
    "osmani airport":     ("Osmani International Airport",    24.9631, 91.8674, 200),
    "zindabazar":         ("Zindabazar, Sylhet",              24.8949, 91.8760, 100),
    "ambarkhana":         ("Ambarkhana, Sylhet",              24.8960, 91.8784, 150),
    "upashahar":          ("Upashahar, Sylhet",               24.9044, 91.8643, 200),

    # Dhaka general landmarks
    "dhaka station":      ("Dhaka Railway Station, Kamalapur", 23.7330, 90.4270, 150),
    "kamalapur":          ("Kamalapur Railway Station",        23.7330, 90.4270, 150),
    "gulshan 1":          ("Gulshan-1 Circle, Dhaka",         23.7806, 90.4152, 150),
    "gulshan 2":          ("Gulshan-2 Circle, Dhaka",         23.7948, 90.4145, 150),
    "dhanmondi 32":       ("Dhanmondi-32, Dhaka",             23.7461, 90.3742, 100),
    "hatirjheel":         ("Hatirjheel, Dhaka",               23.7614, 90.4098, 300),
    "kawran bazar":       ("Kawran Bazar, Dhaka",             23.7514, 90.3930, 150),
    "farmgate":           ("Farmgate, Dhaka",                 23.7577, 90.3870, 100),
    "sadarghat":          ("Sadarghat Launch Terminal",        23.7104, 90.4074, 100),
    "dhaka medical":      ("Dhaka Medical College Hospital",  23.7229, 90.3952, 100),
    "dmch":               ("Dhaka Medical College Hospital",  23.7229, 90.3952, 100),
    "mitford":            ("Mitford Hospital, Dhaka",         23.7170, 90.3981, 100),
    "shahajalal":         ("Hazrat Shahjalal Intl Airport",   23.8434, 90.3979, 300),
    "uttara sector 3":    ("Uttara Sector-3, Dhaka",          23.8694, 90.3850, 200),
    "uttara sector 7":    ("Uttara Sector-7, Dhaka",          23.8759, 90.3795, 200),
    "tongi":              ("Tongi, Gazipur",                  23.8927, 90.3987, 400),

    # Sirajganj
    "sirajganj river":    ("Sirajganj River Bank",            24.4490, 89.6950, 200),
    "enayetpur":          ("Enayetpur, Sirajganj",            24.3747, 89.6850, 300),
}


# ─────────────────────────────────────────────────────────────────────────────
# AREA CENTROIDS (fallback when no sub-zone is found)
# ─────────────────────────────────────────────────────────────────────────────

AREA_CENTROIDS: dict[str, tuple] = {
    "dhaka":       ("Dhaka (area centroid)",   23.7808, 90.4142, 5000),
    "mirpur":      ("Mirpur (area centroid)",  23.8100, 90.3680, 3000),
    "sylhet":      ("Sylhet (area centroid)",  24.8949, 91.8687, 3000),
    "chittagong":  ("Chittagong (centroid)",   22.3384, 91.8317, 5000),
    "barishal":    ("Barishal (centroid)",     22.7010, 90.3535, 4000),
    "sirajganj":   ("Sirajganj (centroid)",    24.4534, 89.7007, 3000),
    "sunamganj":   ("Sunamganj (centroid)",    25.0715, 91.3953, 4000),
    "kurigram":    ("Kurigram (centroid)",     25.8074, 89.6357, 4000),
    "gaibandha":   ("Gaibandha (centroid)",    25.3286, 89.5284, 4000),
    "jamalpur":    ("Jamalpur (centroid)",     24.9000, 89.9378, 4000),
    "bogra":       ("Bogra (centroid)",        24.8520, 89.3692, 4000),
    "narayanganj": ("Narayanganj (centroid)",  23.6238, 90.4986, 3000),
    "cox's bazar": ("Cox's Bazar (centroid)",  21.4272, 92.0058, 5000),
    "rangpur":     ("Rangpur (centroid)",      25.7439, 89.2752, 4000),
    "rajshahi":    ("Rajshahi (centroid)",     24.3636, 88.6241, 4000),
    "khulna":      ("Khulna (centroid)",       22.8456, 89.5403, 4000),
    "mymensingh":  ("Mymensingh (centroid)",   24.7471, 90.4203, 4000),
    "comilla":     ("Comilla (centroid)",      23.4607, 91.1809, 4000),
    "noakhali":    ("Noakhali (centroid)",     22.8696, 91.0996, 4000),
}


# ─────────────────────────────────────────────────────────────────────────────
# RESOLUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[।,\.!?;:()\[\]\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_number(text: str) -> Optional[str]:
    """Extract trailing number, e.g. 'mirpur 10' → '10'"""
    m = re.search(r"\b(\d{1,2})\b", text)
    return m.group(1) if m else None


class LocationResolver:
    """
    Resolves a (raw_message, coarse_location) pair to an exact coordinate.

    Usage:
        resolver = LocationResolver()
        loc = resolver.resolve(
            raw_message="Mirpur e pani utheche, bari dube jacche",
            coarse_location="Mirpur, Dhaka"
        )
        print(loc.latitude, loc.longitude, loc.confidence)
    """

    def resolve(
        self,
        raw_message: str,
        coarse_location: str,
    ) -> ResolvedLocation:
        """
        3-layer pipeline:
          1. Landmark / sector keyword in raw message
          2. Sub-zone keyword in coarse location
          3. Fallback to area centroid + uncertainty flag
        """
        msg_norm = _normalise(raw_message)
        loc_norm = _normalise(coarse_location)
        combined = msg_norm + " " + loc_norm

        # ── Layer 1: Landmark / sector number match ────────────────────────
        result = self._try_landmark(combined)
        if result:
            return ResolvedLocation(
                display_name=result[0],
                latitude=result[1],
                longitude=result[2],
                confidence=0.92,
                resolution_method="landmark",
                uncertainty_radius_m=result[3],
                raw_input=coarse_location,
                needs_followup=False,
            )

        # ── Layer 2: Sub-zone keyword match ───────────────────────────────
        result = self._try_subzone(combined, loc_norm)
        if result:
            return ResolvedLocation(
                display_name=result[0],
                latitude=result[1],
                longitude=result[2],
                confidence=0.75,
                resolution_method="subzone_keyword",
                uncertainty_radius_m=result[3],
                raw_input=coarse_location,
                needs_followup=False,
            )

        # ── Layer 3: Area centroid fallback ───────────────────────────────
        result = self._try_centroid(loc_norm)
        if result:
            return ResolvedLocation(
                display_name=result[0],
                latitude=result[1],
                longitude=result[2],
                confidence=0.40,
                resolution_method="area_centroid",
                uncertainty_radius_m=result[3],
                raw_input=coarse_location,
                needs_followup=True,   # ← Tell Agent 2 to ask follow-up
            )

        # ── Last resort: return Dhaka centre ──────────────────────────────
        return ResolvedLocation(
            display_name="Unknown (defaulting to Dhaka centre)",
            latitude=23.7808,
            longitude=90.4142,
            confidence=0.10,
            resolution_method="default_fallback",
            uncertainty_radius_m=10000,
            raw_input=coarse_location,
            needs_followup=True,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _try_landmark(self, combined: str) -> Optional[tuple]:
        # Longest match first to prefer "mirpur 10 circle" over "mirpur 10"
        for key in sorted(LANDMARK_DB.keys(), key=len, reverse=True):
            if key in combined:
                return LANDMARK_DB[key]
        return None

    def _try_subzone(self, combined: str, loc_norm: str) -> Optional[tuple]:
        """Match area name and pick the first/most prominent sub-zone."""
        for area_key, subzones in SUBZONE_DB.items():
            if area_key in combined or area_key in loc_norm:
                # Return the first entry (most prominent sub-zone) as best guess
                # In production Agent 2 could ask "which sector?" for Mirpur
                return subzones[0]
        return None

    def _try_centroid(self, loc_norm: str) -> Optional[tuple]:
        for area_key, centroid in AREA_CENTROIDS.items():
            if area_key in loc_norm:
                return centroid
        return None

    def get_all_subzones(self, area_name: str) -> list[dict]:
        """
        Return all known sub-zones for an area — used by Agent 2 when
        it needs to present a follow-up question to the victim.
        """
        key = area_name.lower().strip()
        subzones = SUBZONE_DB.get(key, [])
        return [
            {"name": s[0], "lat": s[1], "lon": s[2], "uncertainty_m": s[3]}
            for s in subzones
        ]


# Singleton instance for import
resolver = LocationResolver()
