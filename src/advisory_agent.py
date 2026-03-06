"""
advisory_agent.py
Agentic AI Farm Advisory System — Milestone 2.
"""

from __future__ import annotations
import re
from typing import Any

AGRONOMY_KB: dict[str, str] = {
    "irrigation": (
        "FAO recommends deficit irrigation strategies when water is scarce. "
        "Drip irrigation can improve water-use efficiency by 30-50% over flood irrigation. "
        "Critical stages: germination, flowering, and grain-filling."
    ),
    "fertilizer": (
        "ICAR recommends soil-test-based fertilizer application. "
        "Balanced NPK is essential. Split N application reduces losses. "
        "Organic compost improves long-term soil health."
    ),
    "pest_management": (
        "IPM combines biological controls, resistant varieties, cultural practices, "
        "and judicious pesticide use. Economic thresholds should guide decisions."
    ),
    "heat_stress": (
        "Heat stress above 35 degrees reduces photosynthesis and pollen viability. "
        "Irrigate during peak heat, use mulch, select heat-tolerant varieties."
    ),
    "drought_management": (
        "Drought-tolerant varieties maintain 70-80% yield under moisture stress. "
        "Conservation agriculture increases soil water retention."
    ),
    "soil_health": (
        "Healthy soil organic carbon supports water retention and nutrient cycling. "
        "Crop rotation and cover cropping improve long-term soil health."
    ),
    "variety_selection": (
        "Selecting a high-yielding variety adapted to the local agro-climatic zone "
        "is one of the highest-impact decisions a farmer can make. "
        "Use certified seeds from accredited sources."
    ),
    "seasonal_planning": (
        "Sowing at the recommended phenological window maximises yield potential. "
        "Use FAO GIEWS or NOAA crop calendars for guidance."
    ),
}

REQUIRED_SECTIONS = [
    "### 1. Crop and Field Summary",
    "### 2. Yield Prediction Interpretation",
    "### 3. Identified Risk Factors",
    "### 4. Recommended Farming Actions",
    "### 5. Supporting Knowledge",
    "### 6. Disclaimer",
]


def _retrieve_knowledge(
    rainfall: float,
    temperature: float,
    pesticides: float,
    risks: list[dict[str, str]],
) -> str:
    """Retrieve relevant agronomy snippets (keyword-based RAG step)."""
    selected: list[str] = []
    used: set[str] = set()

    selected.append(f"[Variety Selection] {AGRONOMY_KB['variety_selection']}")
    used.add("variety_selection")
    selected.append(f"[Seasonal Planning] {AGRONOMY_KB['seasonal_planning']}")
    used.add("seasonal_planning")

    if rainfall < 500:
        selected.append(f"[Drought Management] {AGRONOMY_KB['drought_management']}")
        used.add("drought_management")
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")
        used.add("irrigation")
    elif rainfall > 2000:
        selected.append(f"[Soil Health] {AGRONOMY_KB['soil_health']}")
        used.add("soil_health")

    if temperature > 33:
        selected.append(f"[Heat Stress Management] {AGRONOMY_KB['heat_stress']}")
        used.add("heat_stress")

    if pesticides < 50:
        selected.append(f"[Pest Management] {AGRONOMY_KB['pest_management']}")
        used.add("pest_management")

    if "fertilizer" not in used:
        selected.append(f"[Fertilizer Management] {AGRONOMY_KB['fertilizer']}")
    if "irrigation" not in used:
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")

    return "

".join(selected)
