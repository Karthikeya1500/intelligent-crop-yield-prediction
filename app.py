import io
import json
from fpdf import FPDF
import pickle
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title='Intelligent Crop Yield Prediction & Advisory', page_icon=None, layout='wide', initial_sidebar_state='expanded')
st.markdown('\n<style>\n  @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\');\n\n  html, body, [class*="css"] { font-family: \'Inter\', sans-serif; }\n\n  .main-header {\n      font-size: 2.4rem;\n      font-weight: 700;\n      background: linear-gradient(135deg, #1B5E20, #388E3C, #66BB6A);\n      -webkit-background-clip: text;\n      -webkit-text-fill-color: transparent;\n      text-align: center;\n      padding: 0.6rem 0 0.2rem;\n  }\n  .sub-header {\n      font-size: 1.05rem;\n      color: #78909C;\n      text-align: center;\n      margin-bottom: 1.5rem;\n  }\n  .metric-card {\n      background: linear-gradient(135deg, #E8F5E9, #C8E6C9);\n      padding: 1.2rem;\n      border-radius: 12px;\n      text-align: center;\n      border-left: 4px solid #2E7D32;\n  }\n  .risk-high   { /* High severity — red, high contrast for dark/light mode */ border-left: 6px solid #B71C1C; background:#C62828; color:#ffffff; padding:0.9rem 1.2rem; border-radius:8px; margin:8px 0; }\n  .risk-high strong { color:#FFE0E0; }\n  .risk-medium { /* Medium severity — orange */ border-left: 6px solid #E65100; background:#EF6C00; color:#ffffff; padding:0.9rem 1.2rem; border-radius:8px; margin:8px 0; }\n  .risk-medium strong { color:#FFF3E0; }\n  .risk-low    { /* Low severity — green */ border-left: 6px solid #1B5E20; background:#2E7D32; color:#ffffff; padding:0.9rem 1.2rem; border-radius:8px; margin:8px 0; }\n  .risk-low strong { color:#E8F5E9; }\n\n  .yield-badge-low    { background:#FFCDD2; color:#B71C1C; padding:4px 14px; border-radius:20px; font-weight:600; }\n  .yield-badge-medium { background:#FFE0B2; color:#E65100; padding:4px 14px; border-radius:20px; font-weight:600; }\n  .yield-badge-high   { background:#C8E6C9; color:#1B5E20; padding:4px 14px; border-radius:20px; font-weight:600; }\n\n  div.stButton > button {\n      background: linear-gradient(135deg, #2E7D32, #43A047);\n      color: white;\n      border-radius: 8px;\n      padding: 0.5rem 2rem;\n      font-weight: 600;\n      border: none;\n      transition: opacity 0.2s;\n  }\n  div.stButton > button:hover { opacity: 0.88; }\n\n  .step-badge {\n      display:inline-block;\n      background:#2E7D32;\n      color:white;\n      border-radius:50%;\n      width:26px; height:26px;\n      line-height:26px;\n      text-align:center;\n      font-weight:700;\n      margin-right:8px;\n      font-size:0.85rem;\n  }\n</style>\n', unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open('models/crop_yield_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    return (model, label_encoders, scaler, model_columns, metrics)
try:
    (model, label_encoders, scaler, model_columns, metrics) = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
st.sidebar.markdown('## Navigation')
page = st.sidebar.radio('Go to', ['Predict Yield', 'Farm Advisory', 'Data Explorer', 'Model Performance'], label_visibility='collapsed')
st.sidebar.divider()
try:
    groq_api_key = st.secrets.get('GROQ_API_KEY', os.environ.get('GROQ_API_KEY', ''))
except Exception:
    groq_api_key = os.environ.get('GROQ_API_KEY', '')
if page == 'Predict Yield':
    st.markdown('<p class="main-header">Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter agricultural parameters to predict crop yield (hg/ha)</p>', unsafe_allow_html=True)
    if not model_loaded:
        st.error('Model artifacts not found. Run `python src/train_model.py` first.')
        st.stop()
    st.divider()
    (col1, col2) = st.columns(2)
    with col1:
        st.subheader('Crop & Region')
        crop_options = list(label_encoders['Item'].classes_)
        selected_crop = st.selectbox('Crop Type', crop_options, index=0, key='p1_crop')
        area_options = list(label_encoders['Area'].classes_)
        selected_area = st.selectbox('Country / Region', area_options, index=0, key='p1_area')
        selected_year = st.number_input('Year', min_value=1960, max_value=2030, value=2024, step=1, key='p1_year')
    with col2:
        st.subheader('Environmental Factors')
        rainfall = st.number_input('Average Rainfall (mm/year)', min_value=0.0, max_value=5000.0, value=1000.0, step=10.0, key='p1_rain')
        pesticides = st.number_input('Pesticides Used (tonnes)', min_value=0.0, max_value=500000.0, value=1000.0, step=100.0, key='p1_pest')
        avg_temp = st.number_input('Average Temperature (C)', min_value=-10.0, max_value=50.0, value=25.0, step=0.5, key='p1_temp')
    st.divider()
    if st.button('Predict Yield', use_container_width=True, key='btn_predict'):
        crop_encoded = label_encoders['Item'].transform([selected_crop])[0]
        area_encoded = label_encoders['Area'].transform([selected_area])[0]
        input_data = {}
        for col in model_columns:
            if col == 'Area':
                input_data[col] = area_encoded
            elif col == 'Item':
                input_data[col] = crop_encoded
            elif col == 'Year':
                input_data[col] = selected_year
            elif col == 'average_rain_fall_mm_per_year':
                input_data[col] = rainfall
            elif col == 'pesticides_tonnes':
                input_data[col] = pesticides
            elif col == 'avg_temp':
                input_data[col] = avg_temp
            else:
                input_data[col] = 0
        input_df = pd.DataFrame([input_data])
        numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        cols_to_scale = [c for c in numeric_cols if c in input_df.columns]
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
        prediction = model.predict(input_df)[0]
        st.success(f'### Predicted Yield: **{prediction:,.2f} hg/ha**')
        (m1, m2, m3) = st.columns(3)
        with m1:
            st.metric('Crop', selected_crop)
        with m2:
            st.metric('Region', selected_area)
        with m3:
            st.metric('Predicted Yield', f'{prediction:,.0f} hg/ha')
        st.session_state['last_prediction'] = {'crop': selected_crop, 'area': selected_area, 'year': selected_year, 'rainfall': rainfall, 'pesticides': pesticides, 'avg_temp': avg_temp, 'predicted_yield': prediction}
        st.info('Switch to **Farm Advisory** in the sidebar to generate an AI advisory report.')
elif page == 'Farm Advisory':
    st.markdown('<p class="main-header">Agentic Farm Advisory System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered crop management recommendations based on yield prediction & risk analysis</p>', unsafe_allow_html=True)
    st.divider()
    if not model_loaded:
        st.error('Model artifacts not found. Run `python src/train_model.py` first.')
        st.stop()
    last = st.session_state.get('last_prediction', {})
    with st.expander('Farm Input Parameters', expanded=True):
        (col1, col2) = st.columns(2)
        with col1:
            st.subheader('Crop & Region')
            crop_options = list(label_encoders['Item'].classes_)
            default_crop_idx = crop_options.index(last['crop']) if last.get('crop') in crop_options else 0
            adv_crop = st.selectbox('Crop Type', crop_options, index=default_crop_idx, key='adv_crop')
            area_options = list(label_encoders['Area'].classes_)
            default_area_idx = area_options.index(last['area']) if last.get('area') in area_options else 0
            adv_area = st.selectbox('Country / Region', area_options, index=default_area_idx, key='adv_area')
            adv_year = st.number_input('Year', min_value=1960, max_value=2030, value=int(last.get('year', 2024)), step=1, key='adv_year')
        with col2:
            st.subheader('Environmental Factors')
            adv_rainfall = st.number_input('Average Rainfall (mm/year)', min_value=0.0, max_value=5000.0, value=float(last.get('rainfall', 1000.0)), step=10.0, key='adv_rain')
            adv_pesticides = st.number_input('Pesticides Used (tonnes)', min_value=0.0, max_value=500000.0, value=float(last.get('pesticides', 1000.0)), step=100.0, key='adv_pest')
            adv_temp = st.number_input('Average Temperature (C)', min_value=-10.0, max_value=50.0, value=float(last.get('avg_temp', 25.0)), step=0.5, key='adv_temp')
            adv_days = st.number_input('Days Since Planting', min_value=0, max_value=360, value=30, step=1, key='adv_days')
    run_advisory = st.button('Generate Farm Advisory Report', use_container_width=True, key='btn_advisory')
    if run_advisory:
        crop_encoded = label_encoders['Item'].transform([adv_crop])[0]
        area_encoded = label_encoders['Area'].transform([adv_area])[0]
        input_data = {}
        for col in model_columns:
            if col == 'Area':
                input_data[col] = area_encoded
            elif col == 'Item':
                input_data[col] = crop_encoded
            elif col == 'Year':
                input_data[col] = adv_year
            elif col == 'average_rain_fall_mm_per_year':
                input_data[col] = adv_rainfall
            elif col == 'pesticides_tonnes':
                input_data[col] = adv_pesticides
            elif col == 'avg_temp':
                input_data[col] = adv_temp
            else:
                input_data[col] = 0
        input_df = pd.DataFrame([input_data])
        numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        cols_to_scale = [c for c in numeric_cols if c in input_df.columns]
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
        predicted_yield = model.predict(input_df)[0]
        feature_importance: dict = metrics.get('feature_importance', {})
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.risk_analyzer import analyze_risks
        risks = analyze_risks(crop=adv_crop, area=adv_area, year=adv_year, rainfall=adv_rainfall, temperature=adv_temp, pesticides=adv_pesticides, predicted_yield=predicted_yield, feature_importance=feature_importance)
        st.divider()
        st.subheader('Advisory Generation Progress')
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown('<span class="step-badge">1</span> **Yield Prediction** — ML model complete', unsafe_allow_html=True)
        progress_bar.progress(25)
        status_text.markdown('<span class="step-badge">2</span> **Risk Analysis** — Evaluating farm conditions...', unsafe_allow_html=True)
        progress_bar.progress(50)
        status_text.markdown('<span class="step-badge">3</span> **Knowledge Retrieval** — Fetching agronomy guidelines...', unsafe_allow_html=True)
        progress_bar.progress(75)
        status_text.markdown('<span class="step-badge">4</span> **Report Generation** — ' + ('Calling LangGraph Agents & Llama-3...' if groq_api_key else 'Missing API Key. Executing fallback logic...'), unsafe_allow_html=True)
        from src.advisory_agent import run_advisory_agent
        result = run_advisory_agent(crop=adv_crop, area=adv_area, year=adv_year, rainfall=adv_rainfall, temperature=adv_temp, pesticides=adv_pesticides, predicted_yield=predicted_yield, feature_importance=feature_importance, risks=risks, days_since_planting=adv_days, api_key=groq_api_key or '')
        progress_bar.progress(100)
        status_text.markdown('**Advisory report generated successfully.**')
        st.divider()
        st.subheader('Prediction Summary')
        (c1, c2, c3, c4) = st.columns(4)
        yield_cat = result['yield_category']
        badge_class = {'Low': 'yield-badge-low', 'Medium': 'yield-badge-medium', 'High': 'yield-badge-high'}.get(yield_cat, 'yield-badge-medium')
        with c1:
            st.metric('Crop', adv_crop)
        with c2:
            st.metric('Region', adv_area)
        with c3:
            st.metric('Predicted Yield', f'{predicted_yield:,.0f} hg/ha')
        with c4:
            st.markdown(f"**Yield Category**<br><span class='{badge_class}'>{yield_cat}</span>", unsafe_allow_html=True)
        st.divider()
        st.subheader('Identified Risk Factors')
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        sorted_risks = sorted(risks, key=lambda r: severity_order.get(r['severity'], 3))
        for risk in sorted_risks:
            sev = risk['severity'].lower()
            st.markdown(f"""<div class="risk-{sev}"><strong>[{risk['severity']}] {risk['factor']}</strong><br>{risk['description']}</div>""", unsafe_allow_html=True)
        st.divider()
        st.subheader('Farm Advisory Report')
        st.markdown(result['report'])
        st.divider()
        st.subheader('Export Advisory Report')
        (export_col1, export_col2, export_col3) = st.columns(3)
        with export_col1:
            md_content = f"# Farm Advisory Report\n**Crop:** {adv_crop}  \n**Location:** {adv_area}  \n**Year:** {adv_year}  \n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  \n**Predicted Yield:** {predicted_yield:,.0f} hg/ha ({yield_cat})\n\n---\n\n{result['report']}\n"
            st.download_button(label='Download as Markdown (.md)', data=md_content.encode('utf-8'), file_name=f"farm_advisory_{adv_crop.replace(' ', '_')}_{adv_year}.md", mime='text/markdown', use_container_width=True, key='dl_md')
        with export_col2:
            plain = result['report'].replace('**', '').replace('*', '').replace('###', '')
            txt_content = f"FARM ADVISORY REPORT\n{'=' * 36}\nCrop:            {adv_crop}\nLocation:        {adv_area}\nYear:            {adv_year}\nPredicted Yield: {predicted_yield:,.0f} hg/ha ({yield_cat})\nGenerated:       {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n{plain}\n"
            st.download_button(label='Download as Text (.txt)', data=txt_content.encode('utf-8'), file_name=f"farm_advisory_{adv_crop.replace(' ', '_')}_{adv_year}.txt", mime='text/plain', use_container_width=True, key='dl_txt')
        with export_col3:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Helvetica', size=11)
                pdf.set_font('Helvetica', 'B', 16)
                pdf.cell(190, 10, txt='Farm Advisory Report', ln=True, align='C')
                pdf.ln(5)
                pdf.set_font('Helvetica', 'I', 11)
                pdf.cell(190, 6, txt=f'Crop: {adv_crop}  |  Location: {adv_area}', ln=True)
                pdf.cell(190, 6, txt=f"Yield: {predicted_yield:,.0f} hg/ha  | Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
                pdf.ln(5)
                pdf.set_font('Helvetica', size=10)
                lines = result['report'].split('\n')
                import re
                for line in lines:
                    line = line.replace('**', '').replace('*', '-')
                    line = re.sub('([^\\s]{80,})', '\\1 ', line)
                    line = re.sub('[-=]{8,}', '---', line)
                    try:
                        pdf.multi_cell(0, 6, txt=line)
                    except Exception:
                        pdf.multi_cell(0, 6, txt='[Unrenderable formatting removed]')
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button(label='Download as PDF (.pdf)', data=pdf_bytes, file_name=f"farm_advisory_{adv_crop.replace(' ', '_')}_{adv_year}.pdf", mime='application/pdf', use_container_width=True, key='dl_pdf')
            except Exception as e:
                st.error(f'PDF Generation Failed: {e}')
        st.session_state['last_advisory'] = result
elif page == 'Data Explorer':
    st.markdown('<p class="main-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload and explore crop yield data</p>', unsafe_allow_html=True)
    st.divider()
    uploaded_file = st.file_uploader('Upload a CSV file (or the built-in dataset will be used)', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f'Uploaded file loaded: {df.shape[0]} rows x {df.shape[1]} columns')
    else:
        try:
            df = pd.read_csv('data/yield_df.csv')
            st.info('Using built-in dataset: data/yield_df.csv')
        except FileNotFoundError:
            st.warning('No dataset found. Upload a CSV file to explore.')
            st.stop()
    st.subheader('Dataset Overview')
    (o1, o2, o3, o4) = st.columns(4)
    with o1:
        st.metric('Rows', df.shape[0])
    with o2:
        st.metric('Columns', df.shape[1])
    with o3:
        st.metric('Missing Values', int(df.isnull().sum().sum()))
    with o4:
        st.metric('Numeric Cols', len(df.select_dtypes(include=[np.number]).columns))
    st.subheader('Data Preview')
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader('Statistical Summary')
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader('Distribution Plots')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox('Select a numeric column', numeric_cols)
        (fig, axes) = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df[selected_col].dropna(), bins=30, color='#2E7D32', edgecolor='white', alpha=0.85)
        axes[0].set_title(f'Distribution of {selected_col}')
        axes[0].set_xlabel(selected_col)
        axes[0].set_ylabel('Frequency')
        axes[1].boxplot(df[selected_col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='#C8E6C9'))
        axes[1].set_title(f'Box Plot of {selected_col}')
        axes[1].set_ylabel(selected_col)
        plt.tight_layout()
        st.pyplot(fig)
    if len(numeric_cols) > 1:
        st.subheader('Correlation Heatmap')
        (fig_corr, ax_corr) = plt.subplots(figsize=(8, 6))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='Greens', fmt='.2f', ax=ax_corr, linewidths=0.5)
        ax_corr.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig_corr)
elif page == 'Model Performance':
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluation metrics and feature importance analysis</p>', unsafe_allow_html=True)
    st.divider()
    if not model_loaded:
        st.error('Model artifacts not found. Run `python src/train_model.py` first.')
        st.stop()
    best_model_name = metrics.get('best_model', 'N/A')
    st.success(f'**Best Model: {best_model_name}**')
    st.subheader('Model Comparison')
    model_names = [k for k in metrics.keys() if k not in ('best_model', 'feature_importance')]
    comparison_data = []
    for name in model_names:
        m = metrics[name]
        comparison_data.append({'Model': name, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2 Score': m['R2']})
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.subheader('R2 Score Comparison')
    (fig_r2, ax_r2) = plt.subplots(figsize=(8, 4))
    colors = ['#2E7D32' if n == best_model_name else '#81C784' for n in comparison_df['Model']]
    bars = ax_r2.barh(comparison_df['Model'], comparison_df['R2 Score'], color=colors, edgecolor='white')
    ax_r2.set_xlabel('R2 Score')
    ax_r2.set_title('Model R2 Score Comparison')
    for (bar, val) in zip(bars, comparison_df['R2 Score']):
        ax_r2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_r2)
    feature_importance = metrics.get('feature_importance', {})
    if feature_importance:
        st.subheader('Feature Importance')
        fi_df = pd.DataFrame(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True), columns=['Feature', 'Importance'])
        fi_df['Feature'] = fi_df['Feature'].str.replace('_', ' ').str.title()
        (fig_fi, ax_fi) = plt.subplots(figsize=(8, 5))
        ax_fi.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='#66BB6A', edgecolor='white')
        ax_fi.set_xlabel('Importance Score')
        ax_fi.set_title('Yield-Driving Feature Importance')
        plt.tight_layout()
        st.pyplot(fig_fi)
        with st.expander('Feature Importance Table'):
            fi_df['Importance %'] = (fi_df['Importance'] * 100).round(2).astype(str) + '%'
            st.dataframe(fi_df[['Feature', 'Importance %']], use_container_width=True, hide_index=True)
    else:
        st.info('Feature importance data not available.')
st.divider()
st.markdown('<center><small>Intelligent Crop Yield Prediction &amp; Agentic Farm Advisory System &nbsp;|&nbsp; Powered by scikit-learn &amp; Groq Llama-3</small></center>', unsafe_allow_html=True)