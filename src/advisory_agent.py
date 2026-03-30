"""
advisory_agent.py

Agentic AI Farm Advisory System — Milestone 2.

Implements a multi-step agentic workflow:
  Step 1 (Retrieve)  — gather context from static agronomy knowledge base
  Step 2 (Reason)    — build a structured prompt with all context
  Step 3 (Generate)  — call Google Gemini free-tier LLM to produce the report
  Step 4 (Validate)  — check output has all required sections; retry once if not
  Step 5 (Return)    — return the final structured advisory report

LLM Backend: Google Gemini (gemini-1.5-flash via google-generativeai SDK — free tier).
Fallback: If no API key is provided, a rule-based template is used.
"""

from __future__ import annotations
import re
import textwrap
from typing import Any

AGRONOMY_KB: dict[str, str] = {
    "irrigation": (
        "FAO recommends deficit irrigation strategies when water is scarce. "
        "Drip irrigation can improve water-use efficiency by 30-50% versus flood irrigation. "
        "Critical growth stages for water: germination, flowering, and grain-filling."
    ),
    "fertilizer": (
        "ICAR recommends soil-test-based fertilizer application. "
        "Balanced NPK nutrition is essential: N for vegetative growth, P for root development, "
        "K for disease resistance and grain quality. Split application of N reduces losses. "
        "Organic matter (compost, green manure) improves soil health long-term."
    ),
    "pest_management": (
        "Integrated Pest Management (IPM) combines biological controls, resistant varieties, "
        "cultural practices (crop rotation, sanitation), and judicious pesticide use. "
        "Economic thresholds should guide pesticide decisions. "
        "WHO recommends avoiding Class I pesticides and using safer alternatives."
    ),
    "heat_stress": (
        "Heat stress above 35 degrees reduces photosynthesis and causes pollen sterility. "
        "Mitigation: irrigate during peak heat, use mulch to reduce soil temperature, "
        "select heat-tolerant varieties, apply foliar potassium to improve heat resilience."
    ),
    "drought_management": (
        "Drought-tolerant varieties can maintain 70-80% yield under moisture stress. "
        "Conservation agriculture (no-till, residue retention) increases soil water retention. "
        "Rainwater harvesting and micro-irrigation are key FAO-recommended strategies."
    ),
    "soil_health": (
        "Healthy soil organic carbon (above 1.5%) supports water retention, nutrient cycling, "
        "and microbial activity. Regular crop rotation, cover cropping, and reduced tillage "
        "improve long-term soil health. Soil pH 6-7 is optimal for most crops."
    ),
    "variety_selection": (
        "Selecting a high-yielding variety (HYV) adapted to the local agro-climatic zone "
        "is one of the highest-impact decisions. Use certified seeds from accredited sources. "
        "CGIAR and national programs (ICAR, CIMMYT) provide location-specific variety recommendations."
    ),
    "seasonal_planning": (
        "Sowing at the recommended phenological window maximises yield potential. "
        "Late sowing often leads to heat/drought coinciding with critical growth stages. "
        "Use of crop calendars (FAO GIEWS, NOAA) helps align planting with optimal conditions."
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
    selected: list[str] = []
    keys_used: set[str] = set()

    selected.append(f"[Variety Selection] {AGRONOMY_KB['variety_selection']}")
    keys_used.add("variety_selection")
    selected.append(f"[Seasonal Planning] {AGRONOMY_KB['seasonal_planning']}")
    keys_used.add("seasonal_planning")

    if rainfall < 500:
        if "drought_management" not in keys_used:
            selected.append(f"[Drought Management] {AGRONOMY_KB['drought_management']}")
            keys_used.add("drought_management")
        if "irrigation" not in keys_used:
            selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")
            keys_used.add("irrigation")
    elif rainfall > 2000:
        if "soil_health" not in keys_used:
            selected.append(f"[Soil Health] {AGRONOMY_KB['soil_health']}")
            keys_used.add("soil_health")

    if temperature > 33:
        if "heat_stress" not in keys_used:
            selected.append(f"[Heat Stress Management] {AGRONOMY_KB['heat_stress']}")
            keys_used.add("heat_stress")

    if pesticides < 50:
        if "pest_management" not in keys_used:
            selected.append(f"[Pest Management] {AGRONOMY_KB['pest_management']}")
            keys_used.add("pest_management")

    if "fertilizer" not in keys_used:
        selected.append(f"[Fertilizer Management] {AGRONOMY_KB['fertilizer']}")
    if "irrigation" not in keys_used:
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")

    return "

".join(selected)


def _build_prompt(crop, area, year, rainfall, temperature, pesticides,
                  predicted_yield, yield_category, yield_explanation,
                  feature_importance_text, risks_text, knowledge_context):
    return (
        "You are an expert agricultural advisor AI trained on agronomy best practices "
        "from FAO and ICAR.

"
        "Your task is to analyze farm data and generate a structured advisory report.

"
        f"Crop: {crop}  |  Location: {area}  |  Year: {year}
"
        f"Rainfall: {rainfall:.1f} mm/year  |  Temperature: {temperature:.1f} C  "
        f"|  Pesticides: {pesticides:,.1f} tonnes
"
        f"Predicted Yield: {predicted_yield:,.0f} hg/ha ({yield_category})
"
        f"Category basis: {yield_explanation}

"
        f"FEATURE IMPORTANCE:
{feature_importance_text}

"
        f"RISK FACTORS:
{risks_text}

"
        f"AGRONOMY KNOWLEDGE:
{knowledge_context}

"
        "Generate report with these exact sections:
"
        + "
".join(REQUIRED_SECTIONS)
        + "

Be specific, practical, actionable, and based on the data provided."
    )


def _generate_with_gemini(api_key: str, prompt: str) -> str:
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


def _fallback_template(crop, area, year, rainfall, temperature, pesticides,
                       predicted_yield, yield_category, yield_explanation,
                       feature_importance_text, risks_text):
    actions = []
    if rainfall < 500:
        actions.append(
            "**Irrigation (Critical):** Install drip or sprinkler irrigation. "
            "Schedule at critical growth stages. Target 30-50% water-use efficiency improvement."
        )
    elif rainfall > 2000:
        actions.append(
            "**Drainage Management:** Install field drainage channels to prevent waterlogging. "
            "Monitor soil moisture and avoid irrigation during wet periods."
        )
    else:
        actions.append(
            "**Supplemental Irrigation:** Rainfall is moderate. Use supplemental irrigation "
            "during dry spells, particularly at critical growth stages."
        )
    actions.append(
        "**Fertilizer Management:** Conduct a soil test before the season. "
        "Apply balanced NPK based on results. Split nitrogen 2-3 doses. "
        "Incorporate organic compost to build soil organic matter."
    )
    if pesticides < 50:
        actions.append(
            "**Integrated Pest Management (IPM):** Low pesticide use — risk of pest outbreaks. "
            "Implement IPM: pheromone traps, natural predators, crop rotation, "
            "selective pesticides only when thresholds are exceeded."
        )
    else:
        actions.append(
            "**Responsible Pesticide Use:** Rotate chemical classes to prevent resistance. "
            "Adopt IPM to reduce dependency. Ensure operator safety (PPE)."
        )
    if temperature > 35:
        actions.append(
            "**Heat Stress Mitigation:** Apply foliar potassium during flowering. "
            "Irrigate in early morning or evening. Select heat-tolerant certified varieties."
        )
    actions.append(
        "**Variety Selection:** Choose a certified High-Yielding Variety adapted to "
        f"the agro-climatic conditions of {area}. Consult state/national seed boards for "
        f"approved varieties of {crop}. Certified seeds typically yield 15-25% more."
    )
    actions_text = "
".join(f"- Action {i+1}: {a}" for i, a in enumerate(actions))

    lines = [
        "### 1. Crop and Field Summary",
        f"- **Crop:** {crop}",
        f"- **Location:** {area}",
        f"- **Year:** {year}",
        f"- **Key conditions:** Rainfall {rainfall:.0f} mm/year, Temperature {temperature:.1f}°C, "
        f"Pesticide use {pesticides:,.0f} tonnes. Predicted yield: {predicted_yield:,.0f} hg/ha.",
        "", "---", "",
        "### 2. Yield Prediction Interpretation",
        f"- **Predicted yield category:** {yield_category}",
        f"- **Explanation:** {yield_explanation}",
        "", "**Top yield-driving factors:**",
        feature_importance_text,
        "", "---", "",
        "### 3. Identified Risk Factors",
        risks_text,
        "", "---", "",
        "### 4. Recommended Farming Actions",
        actions_text,
        "", "---", "",
        "### 5. Supporting Knowledge / Sources",
        "- **FAO:** Crop Water Requirements, Deficit Irrigation Strategies, IPM guidelines.",
        "- **ICAR:** Soil-test-based fertilizer recommendations, HYV seed programs.",
        "- **CGIAR / CIMMYT:** Drought-tolerant variety programs, heat-stress adaptation.",
        "- **WHO Pesticide Hazard Classification:** Safety guidelines for pesticide selection.",
        "", "---", "",
        "### 6. Disclaimer",
        "This is an AI-generated advisory report for informational purposes only. "
        "Farmers should consult local agricultural extension officers before making decisions. "
        "Yield predictions are indicative estimates only.",
    ]
    return "
".join(lines)


def run_advisory_agent(
    crop: str,
    area: str,
    year: int,
    rainfall: float,
    temperature: float,
    pesticides: float,
    predicted_yield: float,
    feature_importance: dict[str, float],
    risks: list[dict[str, str]],
    api_key: str = "",
) -> dict[str, Any]:
    """Run the full agentic advisory workflow and return report + metadata."""
    # Imports here to avoid circular dependency at module load time
    from src.risk_analyzer import (
        classify_yield,
        format_risks_for_prompt,
        format_feature_importance_for_prompt,
    )

    yield_category, yield_explanation = classify_yield(crop, predicted_yield)
    feature_importance_text = format_feature_importance_for_prompt(feature_importance)
    risks_text = format_risks_for_prompt(risks)
    knowledge_context = _retrieve_knowledge(rainfall, temperature, pesticides, risks)

    result: dict[str, Any] = {
        "yield_category": yield_category,
        "yield_explanation": yield_explanation,
        "risks": risks,
        "used_llm": False,
        "error": None,
        "report": "",
    }

    if api_key and api_key.strip():
        prompt = _build_prompt(
            crop=crop, area=area, year=year, rainfall=rainfall,
            temperature=temperature, pesticides=pesticides,
            predicted_yield=predicted_yield, yield_category=yield_category,
            yield_explanation=yield_explanation,
            feature_importance_text=feature_importance_text,
            risks_text=risks_text, knowledge_context=knowledge_context,
        )
        try:
            report_text = _generate_with_gemini(api_key, prompt)
            if not _validate_output(report_text):
                retry = prompt + (
                    "

IMPORTANT: Your previous response was missing required sections. "
                    "Include ALL six sections with exact headers:
" + "
".join(REQUIRED_SECTIONS)
                )
                report_text = _generate_with_gemini(api_key, retry)
            result["report"] = report_text
            result["used_llm"] = True
        except RuntimeError as exc:
            result["error"] = str(exc)
            result["report"] = _fallback_template(
                crop=crop, area=area, year=year, rainfall=rainfall,
                temperature=temperature, pesticides=pesticides,
                predicted_yield=predicted_yield, yield_category=yield_category,
                yield_explanation=yield_explanation,
                feature_importance_text=feature_importance_text, risks_text=risks_text,
            )
    else:
        result["report"] = _fallback_template(
            crop=crop, area=area, year=year, rainfall=rainfall,
            temperature=temperature, pesticides=pesticides,
            predicted_yield=predicted_yield, yield_category=yield_category,
            yield_explanation=yield_explanation,
            feature_importance_text=feature_importance_text, risks_text=risks_text,
        )

    return result
