"""
risk_analyzer.py
Rule-based agricultural risk analyzer.
Evaluates rainfall, temperature, pesticide use, and yield against benchmarks.
"""

from typing import Any

RAINFALL_THRESHOLDS = {
    "very_low":  200,
    "low":       500,
    "optimal":  1500,
    "high":     2500,
    "very_high": 4000,
}

TEMP_THRESHOLDS = {
    "very_cold":  5,
    "cold":      10,
    "optimal_lo": 15,
    "optimal_hi": 30,
    "hot":        35,
    "very_hot":   40,
}

PESTICIDE_HIGH_THRESHOLD = 50_000
PESTICIDE_LOW_THRESHOLD  = 10

# FAO global average yields by crop (hg/ha)
CROP_YIELD_BENCHMARKS: dict[str, float] = {
    "Wheat":        32_000,
    "Rice, paddy":  46_000,
    "Maize":        55_000,
    "Potatoes":    202_000,
    "Soybeans":     27_000,
    "Cassava":     126_000,
    "Sugar cane":  700_000,
    "Sorghum":      15_000,
    "Barley":       29_000,
    "Groundnuts":   17_000,
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
) -> list:
    """Identify agricultural risk factors from input parameters."""
    risks: list = []

    # Rainfall
    if rainfall < RAINFALL_THRESHOLDS["very_low"]:
        risks.append({
            "factor": "Severe Drought Stress",
            "severity": "High",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is critically low. "
                "Irrigation is essential to prevent crop failure."
            ),
        })
    elif rainfall < RAINFALL_THRESHOLDS["low"]:
        risks.append({
            "factor": "Water Deficit",
            "severity": "Medium",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is below the 500 mm threshold. "
                "Supplemental irrigation is recommended."
            ),
        })
    elif rainfall > RAINFALL_THRESHOLDS["very_high"]:
        risks.append({
            "factor": "Flood / Waterlogging Risk",
            "severity": "High",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is extremely high. "
                "Waterlogging can damage roots. Proper drainage is critical."
            ),
        })
    elif rainfall > RAINFALL_THRESHOLDS["high"]:
        risks.append({
            "factor": "Excess Moisture / Leaching",
            "severity": "Medium",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is high. "
                "Nutrient leaching and fungal disease pressure are likely."
            ),
        })

    # Temperature
    if temperature >= TEMP_THRESHOLDS["very_hot"]:
        risks.append({
            "factor": "Extreme Heat Stress",
            "severity": "High",
            "description": (
                f"Average temperature of {temperature:.1f}°C is extremely high. "
                "Photosynthesis and pollen viability are severely impaired above 38°C."
            ),
        })
    elif temperature >= TEMP_THRESHOLDS["hot"]:
        risks.append({
            "factor": "Heat Stress Risk",
            "severity": "Medium",
            "description": (
                f"Average temperature of {temperature:.1f}°C approaches the heat-stress "
                "threshold. Yield losses of 5-15% are common."
            ),
        })
    elif temperature <= TEMP_THRESHOLDS["very_cold"]:
        risks.append({
            "factor": "Frost / Chilling Injury Risk",
            "severity": "High",
            "description": (
                f"Average temperature of {temperature:.1f}°C is near or below freezing. "
                "Most crops will fail. Use frost-tolerant varieties."
            ),
        })
    elif temperature <= TEMP_THRESHOLDS["cold"]:
        risks.append({
            "factor": "Sub-Optimal Temperature",
            "severity": "Low",
            "description": (
                f"Temperature of {temperature:.1f}°C is below the optimal range. "
                "Slower growth and delayed maturity are expected."
            ),
        })

    # Pesticides
    if pesticides <= PESTICIDE_LOW_THRESHOLD:
        risks.append({
            "factor": "Pest / Disease Vulnerability",
            "severity": "Medium",
            "description": (
                f"Very low pesticide use ({pesticides:.1f} tonnes). "
                "Pest outbreaks could significantly reduce yield. Consider IPM practices."
            ),
        })
    elif pesticides > PESTICIDE_HIGH_THRESHOLD:
        risks.append({
            "factor": "Excessive Pesticide Use",
            "severity": "Medium",
            "description": (
                f"Pesticide use of {pesticides:,.0f} tonnes is very high. "
                "Risk of resistance development and soil microbiome disruption."
            ),
        })

    return risks
