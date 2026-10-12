"""Shared naming utilities for crown-center preprocessing."""

from __future__ import annotations

import re


CITY_ALIASES = {
    "abq": "albuquerque", "ana": "anaheim", "arl": "arlington",
    "atl": "atlanta", "bal": "baltimore", "buf": "buffalo",
    "cc": "capecoral", "cos": "coloradosprings", "csp": "coloradosprings",
    "dca": "washingtondc", "dc": "washingtondc", "den": "denver",
    "ggv": "gardengrove", "hnb": "huntingtonbeach", "la": "losangeles",
    "lax": "losangeles", "nyc": "newyork", "sf": "sanfrancisco",
}


def normalize_city_token(value: object) -> str:
    """Return the canonical compact city token used by study products."""
    text = str(value or "").lower().replace("alberquerque", "albuquerque")
    token = re.sub(r"[^a-z0-9]+", "", text)
    return CITY_ALIASES.get(token, token)
