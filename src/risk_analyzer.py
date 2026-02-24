"""
risk_analyzer.py
Rule-based agricultural risk analyzer.
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

    # Rainfall / water availability
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
                "Supplemental irrigation is recommended at critical growth stages."
            ),
        })
    elif rainfall > RAINFALL_THRESHOLDS["very_high"]:
        risks.append({
            "factor": "Flood / Waterlogging Risk",
            "severity": "High",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is extremely high. "
                "Waterlogging can damage roots and spread fungal diseases. "
                "Proper field drainage is critical."
            ),
        })
    elif rainfall > RAINFALL_THRESHOLDS["high"]:
        risks.append({
            "factor": "Excess Moisture / Leaching",
            "severity": "Medium",
            "description": (
                f"Rainfall of {rainfall:.0f} mm/year is above typical crop needs. "
                "Nutrient leaching and disease pressure are likely."
            ),
        })

    return risks
