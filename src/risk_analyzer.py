"""
risk_analyzer.py
Rule-based agricultural risk analyzer (complete version).
"""

from __future__ import annotations
from typing import Any

RAINFALL_THRESHOLDS: dict[str, float] = {
    "very_low":  200,
    "low":       500,
    "optimal":  1500,
    "high":     2500,
    "very_high": 4000,
}

TEMP_THRESHOLDS: dict[str, float] = {
    "very_cold":  5,
    "cold":      10,
    "optimal_lo": 15,
    "optimal_hi": 30,
    "hot":        35,
    "very_hot":   40,
}

PESTICIDE_HIGH_THRESHOLD = 50_000
PESTICIDE_LOW_THRESHOLD  = 10

CROP_YIELD_BENCHMARKS: dict[str, float] = {
    "Wheat":          32_000,
    "Rice, paddy":    46_000,
    "Maize":          55_000,
    "Potatoes":      202_000,
    "Soybeans":       27_000,
    "Cassava":       126_000,
    "Sugar cane":    700_000,
    "Sorghum":        15_000,
    "Barley":         29_000,
    "Groundnuts":     17_000,
    "Sunflower seed": 16_000,
    "Sweet potatoes": 97_000,
}


def analyze_risks(
    crop: str,
    area: str,
    year: int,
    rainfall: float,
    temperature: float,
    pesticides: float,
    predicted_yield: float,
    feature_importance: dict,
) -> list[dict[str, str]]:
    """Identify agricultural risk factors from input parameters."""
    risks: list[dict[str, str]] = []

    # Rainfall
    if rainfall < RAINFALL_THRESHOLDS["very_low"]:
        risks.append({"factor": "Severe Drought Stress", "severity": "High",
            "description": f"Rainfall of {rainfall:.0f} mm/year is critically low. Irrigation essential."})
    elif rainfall < RAINFALL_THRESHOLDS["low"]:
        risks.append({"factor": "Water Deficit", "severity": "Medium",
            "description": f"Rainfall of {rainfall:.0f} mm/year is below the 500 mm threshold. Irrigation recommended."})
    elif rainfall > RAINFALL_THRESHOLDS["very_high"]:
        risks.append({"factor": "Flood / Waterlogging Risk", "severity": "High",
            "description": f"Rainfall of {rainfall:.0f} mm/year is extremely high. Proper drainage is critical."})
    elif rainfall > RAINFALL_THRESHOLDS["high"]:
        risks.append({"factor": "Excess Moisture / Leaching", "severity": "Medium",
            "description": f"Rainfall of {rainfall:.0f} mm/year is high. Nutrient leaching likely."})

    # Temperature
    if temperature >= TEMP_THRESHOLDS["very_hot"]:
        risks.append({"factor": "Extreme Heat Stress", "severity": "High",
            "description": f"Temperature {temperature:.1f}°C severely impairs photosynthesis and pollen viability."})
    elif temperature >= TEMP_THRESHOLDS["hot"]:
        risks.append({"factor": "Heat Stress Risk", "severity": "Medium",
            "description": f"Temperature {temperature:.1f}°C — yield losses of 5-15% are common above 35°C."})
    elif temperature <= TEMP_THRESHOLDS["very_cold"]:
        risks.append({"factor": "Frost / Chilling Injury Risk", "severity": "High",
            "description": f"Temperature {temperature:.1f}°C near freezing. Most crops will fail."})
    elif temperature <= TEMP_THRESHOLDS["cold"]:
        risks.append({"factor": "Sub-Optimal Temperature", "severity": "Low",
            "description": f"Temperature {temperature:.1f}°C below optimal. Slower growth and delayed maturity expected."})

    # Pesticides
    if pesticides <= PESTICIDE_LOW_THRESHOLD:
        risks.append({"factor": "Pest / Disease Vulnerability", "severity": "Medium",
            "description": (f"Very low pesticide use ({pesticides:.1f} tonnes). "
                "Pest outbreaks could reduce yield. Consider Integrated Pest Management (IPM).")})
    elif pesticides > PESTICIDE_HIGH_THRESHOLD:
        risks.append({"factor": "Excessive Pesticide Use", "severity": "Medium",
            "description": (f"Pesticide use of {pesticides:,.0f} tonnes is very high. "
                "Risk of resistance development and soil microbiome damage.")})

    # Yield vs global benchmark
    benchmark = CROP_YIELD_BENCHMARKS.get(crop)
    if benchmark:
        ratio = predicted_yield / benchmark
        if ratio < 0.5:
            risks.append({"factor": "Significantly Below-Average Yield", "severity": "High",
                "description": (f"Predicted yield ({predicted_yield:,.0f} hg/ha) is less than 50% "
                    f"of the global average for {crop} (~{benchmark:,} hg/ha). "
                    "Major agronomic improvements are needed.")})
        elif ratio < 0.75:
            risks.append({"factor": "Below-Average Yield", "severity": "Medium",
                "description": (f"Predicted yield ({predicted_yield:,.0f} hg/ha) is below "
                    f"the global average for {crop} (~{benchmark:,} hg/ha). "
                    "Review soil fertility, variety, and crop management.")})
        elif ratio > 1.3:
            risks.append({"factor": "Exceptionally High Yield — Verify Inputs", "severity": "Low",
                "description": (f"Predicted yield ({predicted_yield:,.0f} hg/ha) exceeds the global "
                    f"average for {crop} by more than 30%. Verify input data accuracy.")})

    # Feature importance risk
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:2]
    for feat, imp in top_features:
        clean = feat.replace("_", " ").lower()
        if imp > 0.4:
            risks.append({"factor": f"High Dependence on {clean.title()}", "severity": "Low",
                "description": (f"'{clean}' contributes {imp*100:.1f}% to yield prediction. "
                    "High reliance on a single factor increases production risk.")})

    if not risks:
        risks.append({"factor": "No Significant Risks Detected", "severity": "Low",
            "description": "Input parameters are within acceptable agronomic ranges for this crop."})

    return risks


def classify_yield(crop: str, predicted_yield: float) -> tuple[str, str]:
    """Classify yield as Low / Medium / High relative to global benchmarks."""
    benchmark = CROP_YIELD_BENCHMARKS.get(crop)
    if benchmark is None:
        median_yield = 50_000
        if predicted_yield < median_yield * 0.6:
            return "Low", f"Predicted yield ({predicted_yield:,.0f} hg/ha) is below typical values."
        elif predicted_yield < median_yield * 1.2:
            return "Medium", f"Predicted yield ({predicted_yield:,.0f} hg/ha) is within a typical range."
        else:
            return "High", f"Predicted yield ({predicted_yield:,.0f} hg/ha) is above typical values."

    ratio = predicted_yield / benchmark
    if ratio < 0.65:
        return ("Low",
            f"Predicted yield ({predicted_yield:,.0f} hg/ha) is significantly below "
            f"the global average for {crop} (~{benchmark:,} hg/ha).")
    elif ratio < 1.1:
        return ("Medium",
            f"Predicted yield ({predicted_yield:,.0f} hg/ha) is below but approaching "
            f"the global average for {crop} (~{benchmark:,} hg/ha).")
    else:
        return ("High",
            f"Predicted yield ({predicted_yield:,.0f} hg/ha) is above "
            f"the global average for {crop} (~{benchmark:,} hg/ha).")


def format_risks_for_prompt(risks: list[dict[str, str]]) -> str:  # noqa: E501
    if not risks:
        return "No significant risks identified."
    lines = []
    for r in risks:
        lines.append(f"[{r['severity']}] {r['factor']}: {r['description']}")
    return "
".join(lines)


def format_feature_importance_for_prompt(feature_importance: dict[str, float]) -> str:  # noqa: E501
    if not feature_importance:
        return "Feature importance data not available."
    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for i, (feat, imp) in enumerate(sorted_fi[:5], 1):
        clean = feat.replace("_", " ").title()
        lines.append(f"  {i}. {clean}: {imp*100:.1f}%")
    return "
".join(lines)
