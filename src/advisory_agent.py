import os
import requests
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
AGRONOMY_KB: dict[str, str] = {'irrigation': 'FAO recommends deficit irrigation strategies when water is scarce. Drip irrigation can improve water-use efficiency by 30-50% versus flood irrigation. Critical growth stages for water: germination, flowering, and grain-filling.', 'fertilizer': 'ICAR recommends soil-test-based fertilizer application. Balanced NPK nutrition is essential: N for vegetative growth, P for root development, K for disease resistance and grain quality. Split application of N reduces losses. Organic matter (compost, green manure) improves soil health long-term.', 'pest_management': 'Integrated Pest Management (IPM) combines biological controls, resistant varieties, cultural practices (crop rotation, sanitation), and judicious pesticide use. Economic thresholds should guide pesticide decisions.', 'heat_stress': 'Heat stress above 35 degrees reduces photosynthesis and causes pollen sterility. Mitigation: irrigate during peak heat, use mulch to reduce soil temperature, select heat-tolerant varieties, apply foliar potassium to improve heat resilience.', 'drought_management': 'Drought-tolerant varieties can maintain 70-80% yield under moisture stress. Conservation agriculture (no-till, residue retention) increases soil water retention. Rainwater harvesting and micro-irrigation are key strategies.', 'soil_health': 'Healthy soil organic carbon (above 1.5%) supports water retention, nutrient cycling, and microbial activity. Regular crop rotation, cover cropping, and reduced tillage improve long-term soil health.', 'variety_selection': 'Selecting a high-yielding variety (HYV) adapted to the local agro-climatic zone is one of the highest-impact decisions. Use certified seeds from accredited sources.', 'seasonal_planning': 'Sowing at the recommended phenological window maximises yield potential. Late sowing often leads to heat/drought coinciding with critical growth stages.'}
REQUIRED_SECTIONS = ['### 1. Crop and Field Summary', '### 2. Yield Prediction Interpretation', '### 3. Identified Risk Factors', '### 4. Recommended Farming Actions', '### 5. Supporting Knowledge', '### 6. Disclaimer']

class AdvisoryState(TypedDict):
    crop: str
    area: str
    year: int
    rainfall: float
    temperature: float
    pesticides: float
    predicted_yield: float
    yield_category: str
    yield_explanation: str
    feature_importance: dict
    risks_base: str
    days_since_planting: int
    api_key: str
    weather_forecast: str
    crop_stage: str
    knowledge_context: str
    llm_risks: str
    report: str
    error: Optional[str]

def weather_node(state: AdvisoryState) -> dict:
    geo_map = {'India': (20.59, 78.96), 'Brazil': (-14.23, -51.92), 'United States': (37.09, -95.71), 'China': (35.86, 104.19), 'Indonesia': (-0.78, 113.92)}
    (lat, lon) = geo_map.get(state['area'], (20.0, 77.0))
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&timezone=auto'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            daily = data.get('daily', {})
            times = daily.get('time', [])[:7]
            temps = daily.get('temperature_2m_max', [])[:7]
            precs = daily.get('precipitation_sum', [])[:7]
            lines = ['**7-Day Regional Forecast (Open-Meteo)**:']
            for (t, temp, p) in zip(times, temps, precs):
                lines.append(f'- {t}: Max {temp}°C, Rain {p}mm')
            return {'weather_forecast': '\n'.join(lines)}
    except Exception as e:
        pass
    return {'weather_forecast': 'Weather forecast unavailable (Using historical baseline only).'}

def crop_stage_node(state: AdvisoryState) -> dict:
    days = state['days_since_planting']
    if days <= 0:
        stage = 'Pre-planting / Soil Preparation'
    elif days < 30:
        stage = 'Seedling / Early Vegetative Stage'
    elif days < 70:
        stage = 'Active Vegetative / Early Flowering Stage'
    elif days < 110:
        stage = 'Flowering / Grain Filling Stage'
    else:
        stage = 'Maturity / Harvest Stage'
    return {'crop_stage': stage}

def knowledge_node(state: AdvisoryState) -> dict:
    rain = state['rainfall']
    temp = state['temperature']
    pest = state['pesticides']
    selected = []
    selected.append(f"[Variety Selection] {AGRONOMY_KB['variety_selection']}")
    selected.append(f"[Seasonal Planning] {AGRONOMY_KB['seasonal_planning']}")
    if rain < 500:
        selected.append(f"[Drought Management] {AGRONOMY_KB['drought_management']}")
        selected.append(f"[Irrigation] {AGRONOMY_KB['irrigation']}")
    elif rain > 2000:
        selected.append(f"[Soil Health] {AGRONOMY_KB['soil_health']}")
    if temp > 33:
        selected.append(f"[Heat Stress] {AGRONOMY_KB['heat_stress']}")
    if pest < 50:
        selected.append(f"[Pest Management] {AGRONOMY_KB['pest_management']}")
    selected.append(f"[Fertilizer] {AGRONOMY_KB['fertilizer']}")
    return {'knowledge_context': '\n\n'.join(selected)}

def risk_node(state: AdvisoryState) -> dict:
    try:
        if not state['api_key'] or not state['api_key'].startswith('gsk_'):
            raise ValueError('No valid Groq API Key provided.')
        llm = ChatGroq(api_key=state['api_key'], model_name='llama-3.1-8b-instant', temperature=0.2)
        sys_msg = SystemMessage(content='You are an expert agricultural risk assessor system.')
        hum_msg = HumanMessage(content=f"Analyze risk for {state['crop']} planted in {state['area']}.\nCurrent Crop Stage: {state['crop_stage']} ({state['days_since_planting']} days since planting)\nWeather Forecast:\n{state['weather_forecast']}\nHistorical Input Risks Identified by ML Model:\n{state['risks_base']}\n\nIdentify 2-3 specific immediate agricultural risks (e.g. heat stress, waterlogging, pests) synthesizing the forecast and current growth stage. Be concise.")
        response = llm.invoke([sys_msg, hum_msg])
        return {'llm_risks': response.content}
    except Exception as e:
        print(f'Risk Gen Error: {e}')
        return {'llm_risks': 'Could not generate LLM specific risks.', 'error': str(e)}

def report_node(state: AdvisoryState) -> dict:
    try:
        if not state['api_key'] or not state['api_key'].startswith('gsk_'):
            raise ValueError('No valid Groq API Key provided.')
        llm = ChatGroq(api_key=state['api_key'], model_name='llama-3.3-70b-versatile', temperature=0.4)
        sys_msg = SystemMessage(content='You are the lead Agricultural Advisor AI. Synthesize all provided data into a professional and actionable Markdown report.')
        prompt = f"Generate a comprehensive farm advisory report for {state['crop']} in {state['area']}.\n\n**Data Context**:\nPredicted Yield: {state['predicted_yield']:,.0f} hg/ha\nYield Category: {state['yield_category']} ({state['yield_explanation']})\nCrop Stage: {state['crop_stage']} (Planted {state['days_since_planting']} days ago)\n7-Day Weather Forecast:\n{state['weather_forecast']}\nHistorical Database Risk Factors:\n{state['risks_base']}\nImmediate Contextual Risks (LLM):\n{state['llm_risks']}\nAgronomy Best Practices Base:\n{state['knowledge_context']}\n\n**REPORT REQUIREMENTS**:\nYou MUST output exactly the following six sections using EXACT headers (H3 Markdown):\n" + '\n'.join(REQUIRED_SECTIONS) + '\n\nEnsure the report is practical, incorporates the weather forecast to give seasonal planning / fertilizer / irrigation advice, and handles the risk analysis directly.'
        response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        return {'report': response.content}
    except Exception as e:
        print(f'Report Gen Error: {e}')
        error_msg = f'Report generation failed: {str(e)}\n\nAPI Key missing or invalid.'
        return {'report': error_msg, 'error': str(e)}

def run_advisory_agent(crop: str, area: str, year: int, rainfall: float, temperature: float, pesticides: float, predicted_yield: float, feature_importance: dict, risks: list, days_since_planting: int, api_key: str) -> dict:
    from src.risk_analyzer import classify_yield, format_risks_for_prompt
    (yield_category, yield_explanation) = classify_yield(crop, predicted_yield)
    risks_text = format_risks_for_prompt(risks)
    workflow = StateGraph(AdvisoryState)
    workflow.add_node('weather', weather_node)
    workflow.add_node('crop_stage', crop_stage_node)
    workflow.add_node('knowledge', knowledge_node)
    workflow.add_node('risk', risk_node)
    workflow.add_node('report', report_node)
    workflow.set_entry_point('weather')
    workflow.add_edge('weather', 'crop_stage')
    workflow.add_edge('crop_stage', 'knowledge')
    workflow.add_edge('knowledge', 'risk')
    workflow.add_edge('risk', 'report')
    workflow.add_edge('report', END)
    app = workflow.compile()
    initial_state = {'crop': crop, 'area': area, 'year': year, 'rainfall': rainfall, 'temperature': temperature, 'pesticides': pesticides, 'predicted_yield': predicted_yield, 'yield_category': yield_category, 'yield_explanation': yield_explanation, 'feature_importance': feature_importance, 'risks_base': risks_text, 'days_since_planting': days_since_planting, 'api_key': api_key, 'error': None}
    final_state = app.invoke(initial_state)
    return {'yield_category': final_state.get('yield_category', ''), 'yield_explanation': final_state.get('yield_explanation', ''), 'report': final_state.get('report', 'Report failed.'), 'error': final_state.get('error'), 'used_llm': True if not final_state.get('error') else False}