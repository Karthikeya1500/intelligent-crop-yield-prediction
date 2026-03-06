"""
advisory_agent.py
Agentic AI Farm Advisory System — Milestone 2.
Implements: Knowledge Retrieval -> Prompt Building -> LLM Generation -> Validation.
"""

from __future__ import annotations
import re
import textwrap
from typing import Any

AGRONOMY_KB: dict[str, str] = {
    "irrigation": (
        "FAO recommends deficit irrigation strategies when water is scarce. "
        "Drip irrigation can improve water-use efficiency by 30-50%. "
        "Critical stages: germination, flowering, and grain-filling."
    ),
    "fertilizer": (
        "ICAR recommends soil-test-based fertilizer application. "
        "Balanced NPK is essential. Split N application reduces losses. "
        "Organic compost improves long-term soil health."
    ),
    "pest_management": (
        "IPM combines biological controls, resistant varieties, cultural practices, "
        "and judicious pesticide use. Use economic thresholds to guide decisions."
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
        "Crop rotation and cover cropping improve soil health long-term."
    ),
    "variety_selection": (
        "Selecting a high-yielding variety adapted to the local agro-climatic zone "
        "is one of the highest-impact decisions. Use certified seeds."
    ),
    "seasonal_planning": (
        "Sowing at the recommended phenological window maximises yield potential. "
        "Use FAO GIEWS crop calendars for guidance."
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


def _retrieve_knowledge(rainfall, temperature, pesticides, risks):
    selected, used = [], set()
    selected.append(f"[Variety Selection] {AGRONOMY_KB['variety_selection']}")
    used.add("variety_selection")
    selected.append(f"[Seasonal Planning] {AGRONOMY_KB['seasonal_planning']}")
    used.add("seasonal_planning")
    if rainfall < 500:
        selected.append(f"[Drought] {AGRONOMY_KB['drought_management']}")
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")
        used.update(["drought_management", "irrigation"])
    elif rainfall > 2000:
        selected.append(f"[Soil Health] {AGRONOMY_KB['soil_health']}")
        used.add("soil_health")
    if temperature > 33:
        selected.append(f"[Heat Stress] {AGRONOMY_KB['heat_stress']}")
        used.add("heat_stress")
    if pesticides < 50:
        selected.append(f"[Pest Management] {AGRONOMY_KB['pest_management']}")
        used.add("pest_management")
    if "fertilizer" not in used:
        selected.append(f"[Fertilizer] {AGRONOMY_KB['fertilizer']}")
    if "irrigation" not in used:
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")
    return "

".join(selected)


def _build_prompt(crop, area, year, rainfall, temperature, pesticides,
                  predicted_yield, yield_category, yield_explanation,
                  feature_importance_text, risks_text, knowledge_context):
    """Build structured prompt for the LLM."""
    prompt = (
        "You are an expert agricultural advisor AI trained on FAO and ICAR best practices.

"
        "Analyze the farm data below and generate a structured advisory report.

"
        f"Crop: {crop}  |  Location: {area}  |  Year: {year}
"
        f"Rainfall: {rainfall:.1f} mm/year  |  Temperature: {temperature:.1f} C  "
        f"|  Pesticides: {pesticides:,.1f} tonnes
"
        f"Predicted Yield: {predicted_yield:,.0f} hg/ha ({yield_category})
"
        f"Reason: {yield_explanation}

"
        f"Feature Importance:
{feature_importance_text}

"
        f"Risk Factors:
{risks_text}

"
        f"Agronomy Knowledge:
{knowledge_context}

"
        "Generate a report with these exact sections:
"
        + "
".join(REQUIRED_SECTIONS)
        + "

Be specific, practical, and based on the data provided."
    )
    return prompt
