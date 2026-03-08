"""
advisory_agent.py
Agentic AI Farm Advisory System — Milestone 2.
"""

from __future__ import annotations
import re
import textwrap
from typing import Any

AGRONOMY_KB: dict[str, str] = {
    "irrigation": (
        "FAO recommends deficit irrigation when water is scarce. "
        "Drip irrigation improves efficiency by 30-50%. "
        "Key stages: germination, flowering, grain-filling."
    ),
    "fertilizer": (
        "ICAR recommends soil-test-based NPK. Split N reduces losses. "
        "Organic compost improves long-term soil health."
    ),
    "pest_management": (
        "IPM: biological controls + resistant varieties + cultural practices + targeted pesticides. "
        "Apply only when economic thresholds are exceeded."
    ),
    "heat_stress": (
        "Heat stress above 35C reduces photosynthesis and pollen viability. "
        "Irrigate during peak heat. Select heat-tolerant varieties."
    ),
    "drought_management": (
        "Drought-tolerant varieties maintain 70-80% yield under moisture stress. "
        "Conservation agriculture improves soil water retention."
    ),
    "soil_health": (
        "Soil organic carbon above 1.5% supports water retention and nutrient cycling. "
        "Crop rotation and cover cropping help."
    ),
    "variety_selection": (
        "High-yielding varieties adapted to the local agro-climatic zone "
        "can improve yields by 15-25%. Use certified seeds."
    ),
    "seasonal_planning": (
        "Sowing at the correct phenological window maximises yield potential. "
        "Use FAO GIEWS crop calendars."
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
    selected.append(f"[Seasonal Planning] {AGRONOMY_KB['seasonal_planning']}")
    used.update(["variety_selection", "seasonal_planning"])
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
    return (
        "You are an expert agricultural advisor AI (FAO/ICAR guidelines).

"
        f"Crop: {crop}  Location: {area}  Year: {year}
"
        f"Rainfall: {rainfall:.1f} mm/yr  Temp: {temperature:.1f}C  "
        f"Pesticides: {pesticides:,.1f} t
"
        f"Predicted Yield: {predicted_yield:,.0f} hg/ha ({yield_category})

"
        f"Feature Importance:
{feature_importance_text}

"
        f"Risks:
{risks_text}

"
        f"Agronomy Context:
{knowledge_context}

"
        "Write a report with these exact sections:
"
        + "
".join(REQUIRED_SECTIONS)
    )


def _generate_with_gemini(api_key: str, prompt: str) -> str:
    """Call Google Gemini free-tier API and return the response text."""
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": 0.4, "top_p": 0.9, "max_output_tokens": 2048},
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        return model.generate_content(prompt).text
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


def _validate_output(text: str) -> bool:
    return all(s.lower() in text.lower() for s in REQUIRED_SECTIONS)
