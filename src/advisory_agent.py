"""
advisory_agent.py
Agentic AI Farm Advisory System — Milestone 2 (work in progress).
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Agronomy Knowledge Base (static context for RAG-style retrieval)
# ---------------------------------------------------------------------------

AGRONOMY_KB: dict[str, str] = {
    "irrigation": (
        "FAO recommends deficit irrigation strategies when water is scarce. "
        "Drip irrigation can improve water-use efficiency by 30-50% over flood irrigation. "
        "Critical growth stages for water: germination, flowering, and grain-filling."
    ),
    "fertilizer": (
        "ICAR recommends soil-test-based fertilizer application. "
        "Balanced NPK nutrition is essential. Split application of N reduces losses. "
        "Organic matter (compost, green manure) improves soil health long-term."
    ),
    "pest_management": (
        "Integrated Pest Management (IPM) combines biological controls, resistant varieties, "
        "cultural practices, and judicious pesticide use. "
        "Economic thresholds should guide pesticide decisions."
    ),
    "heat_stress": (
        "Heat stress above 35 degrees reduces photosynthesis and causes pollen sterility. "
        "Mitigation: irrigate during peak heat, use mulch, select heat-tolerant varieties."
    ),
    "drought_management": (
        "Drought-tolerant varieties can maintain 70-80% yield under moisture stress. "
        "Conservation agriculture increases soil water retention. "
        "Rainwater harvesting and micro-irrigation are key FAO-recommended strategies."
    ),
    "soil_health": (
        "Healthy soil organic carbon supports water retention and nutrient cycling. "
        "Regular crop rotation and cover cropping improve long-term soil health."
    ),
    "variety_selection": (
        "Selecting a high-yielding variety adapted to the local agro-climatic zone "
        "is one of the highest-impact decisions a farmer can make."
    ),
    "seasonal_planning": (
        "Sowing at the recommended phenological window maximises yield potential. "
        "Use of crop calendars helps align planting with optimal conditions."
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
