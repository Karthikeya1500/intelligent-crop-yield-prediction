# Intelligent Crop Yield Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Live Demo:** [Click here to use the Agentic Farm Advisory System](https://intelligent-crop-yield-prediction-irohnwchpirk3hdzn2gt7q.streamlit.app/)

An AI-driven agricultural analytics system that predicts crop yield using historical farm, soil, and weather data. Built with **Scikit-Learn** for machine learning and **Streamlit** for the interactive web interface.

---

## Project Overview

Agricultural yield depends on multiple factors including weather conditions, pesticide usage, and regional characteristics. This system uses supervised machine learning to predict crop yield (in hectograms per hectare) from historical data sourced from FAO and the World Data Bank.

### Key Features

- Multi-model training pipeline — Linear Regression, Decision Tree, and Random Forest
- Automated model selection — best model chosen by R2 score
- Feature importance analysis — identifies key yield-driving factors
- Interactive web UI — predict yield, explore data, and view model performance
- CSV upload support — analyze custom datasets

---

## Project Structure

```
intelligent-crop-yield-prediction/
│
├── .streamlit/
│   └── secrets.toml                # Groq API key and local secrets
│
├── data/
│   └── yield_df.csv                # Dataset (Kaggle)
│
├── models/                         # Generated after training
│   ├── crop_yield_model.pkl        # Best trained model (<50 MB)
│   ├── label_encoders.pkl          # Categorical encoders
│   ├── scaler.pkl                  # Feature scaler
│   ├── model_columns.pkl           # Feature column names
│   └── metrics.json                # Evaluation metrics
│
├── src/
│   ├── advisory_agent.py           # LangChain/LangGraph agentic workflow
│   ├── risk_analyzer.py            # Rule-based agronomic risk engine
│   └── train_model.py              # Training and evaluation pipeline
│
├── .gitignore
├── app.py                          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/intelligent-crop-yield-prediction.git
cd intelligent-crop-yield-prediction
pip install -r requirements.txt
```

### 2. Add the dataset

Download the [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) from Kaggle and place `yield_df.csv` inside the `data/` folder.

### 3. Train the model

```bash
python src/train_model.py
```

This will:
- Preprocess the data (encode, scale, split)
- Train 3 models and evaluate each with MAE, RMSE, R2
- Select the best model and save all artifacts to `models/`

### 4. Launch the app

```bash
streamlit run app.py
```

---

## System Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw CSV     │────>│  Preprocessing   │────>│  Feature        │
│  Dataset     │     │  (Clean, Encode, │     │  Engineering    │
│              │     │   Scale)         │     │  (Label Encode, │
└──────────────┘     └──────────────────┘     │   StandardScale)│
                                              └────────┬────────┘
                                                       │
                     ┌─────────────────────────────────v───────────┐
                     │         Model Training & Evaluation          │
                     │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
                     │  │ Linear   │ │ Decision │ │ Random       │ │
                     │  │ Regress. │ │ Tree     │ │ Forest       │ │
                     │  └──────────┘ └──────────┘ └──────────────┘ │
                     │       MAE / RMSE / R2 Evaluation            │
                     └─────────────────────┬───────────────────────┘
                                           │
                     ┌─────────────────────v───────────────────────┐
                     │            Streamlit Web App                 │
                     │ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐ │
                     │ │ Predict │ │ Farm    │ │ Data  │ │ Model │ │
                     │ │ Yield   │ │ Advisory│ │ Explor│ │ Perf. │ │
                     │ └─────────┘ └─────────┘ └───────┘ └───────┘ │
                     └─────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [Kaggle — Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

| Column | Description |
|--------|-------------|
| `Area` | Country / region |
| `Item` | Crop type (Wheat, Rice, Maize, etc.) |
| `Year` | Year of observation |
| `hg/ha_yield` | Yield in hectograms per hectare (target) |
| `average_rain_fall_mm_per_year` | Annual rainfall in mm |
| `pesticides_tonnes` | Pesticide usage in tonnes |
| `avg_temp` | Average temperature in C |

---

## Models & Evaluation

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| Linear Regression | 62444.31 | 81501.76 | 0.0843 |
| Decision Tree | 9230.17 | 17583.76 | 0.9574 |
| **Random Forest** | **7765.55** | **14256.93** | **0.9720** |

Best model selected: **Random Forest** (n_estimators=20, max_depth=10)

---

## Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data processing
- **Scikit-Learn** — Machine learning models
- **Matplotlib / Seaborn** — Data visualization
- **Streamlit** — Web interface
- **LangChain / LangGraph** — Agentic workflows
- **Groq API / Llama-3** — LLM capabilities
- **FPDF** — PDF generation

---

## License

This project is developed for academic purposes.

Dataset sourced from [FAO](http://www.fao.org/) and [World Data Bank](https://data.worldbank.org/).


## Milestone 2 — Agentic Farm Advisory System

The application now includes an AI-powered Farm Advisory module, integrating Large Language Models and rule-based insights:

- **Risk Analyzer**: Rule-based engine that detects agronomic risks from weather, pesticide, and yield data based on agricultural guidelines.
- **Advisory Agent**: Multi-step agentic workflow using LangChain/LangGraph:
  1. ML model predicts crop yield.
  2. Risk analyzer identifies potential problems.
  3. Knowledge retrieval pulls in targeted agronomy guidelines.
  4. Groq Llama-3 LLM generates a comprehensive, structured advisory report.
- **Export**: Seamlessly share insights by downloading the advisory report as a PDF, Markdown, or plain text directly from the Streamlit UI.

### Setup

Add your Groq API key to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key-here"
```

Get a free key at https://console.groq.com/keys
