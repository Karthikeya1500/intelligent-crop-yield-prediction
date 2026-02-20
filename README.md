# Intelligent Crop Yield Prediction

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
                     │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
                     │  │ Predict  │ │ Data     │ │ Model       │ │
                     │  │ Yield    │ │ Explorer │ │ Performance │ │
                     │  └──────────┘ └──────────┘ └─────────────┘ │
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

---

## License

This project is developed for academic purposes.

Dataset sourced from [FAO](http://www.fao.org/) and [World Data Bank](https://data.worldbank.org/).
