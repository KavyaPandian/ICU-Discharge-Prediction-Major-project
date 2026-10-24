# ICU Discharge Prediction Using Machine Learning

## Project Overview

**ICU Discharge Prediction Using Machine Learning** is a healthcare analytics project designed to analyze Intensive Care Unit (ICU) patient data and develop predictive models to support ICU discharge-related decision-making.

The project uses machine learning and deep learning techniques to process patient information, identify relevant patterns, evaluate predictive performance, and present results through a dashboard.

The system explores multiple predictive approaches, including Random Forest, XGBoost, and Long Short-Term Memory (LSTM) networks.

**Note:** This project is intended for academic research and educational purposes. Its predictions are not a substitute for clinical assessment or medical advice.

## Project Objectives

- Analyze ICU patient data to identify patterns relevant to discharge prediction.
- Apply machine learning and deep learning algorithms to healthcare datasets.
- Compare model performance using appropriate evaluation metrics.
- Visualize model results and performance through a dashboard.
- Explore the potential of data-driven methods to support healthcare decision-making.

## Key Features

- **Data Processing:** Load, organize, and prepare ICU datasets for analysis.
- **Machine Learning:** Develop predictive models using Random Forest and XGBoost.
- **Deep Learning:** Explore LSTM-based modeling for sequential or time-dependent patient data.
- **Model Evaluation:** Assess model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices, where applicable.
- **Data Visualization:** Present relevant results using charts and a dashboard.
- **Model Persistence:** Save trained models and preprocessing objects for subsequent use.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** TensorFlow / Keras
- **Data Balancing:** imbalanced-learn (SMOTE)
- **Model Explainability:** SHAP
- **Dashboard:** Python dashboard application (refer to `dashboard.py` for the framework used)

## Project Structure

```text
ICU_Discharge_Prediction/
│
├── main.py
├── dashboard.py
├── ADMISSIONS.csv
├── CHARTEVENTS.csv
├── D_ITEMS.csv
├── ICUSTAYS.csv
├── INPUTEVENTS_MV.csv
├── LABEVENTS.csv
├── OUTPUTEVENTS.csv
├── PATIENTS.csv
│
├── confusion_matrix.png
├── icu_features_per_step.pkl
├── icu_features.pkl
├── icu_lstm_model.keras
├── icu_pad_size.pkl
├── icu_randomforest_model.pkl
├── icu_scaler.pkl
├── icu_timesteps.pkl
├── icu_xgboost_model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
```

*This structure is based on the current project files. Update it if your implementation changes.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ICU_Discharge_Prediction.git
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Navigate to the project directory

```bash
cd ICU_Discharge_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages required by the project:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib imbalanced-learn tensorflow xgboost
```

Install any additional dashboard dependencies required by `dashboard.py`.

### 5. Run the project

Run the main Python script:

```bash
python main.py
```

To launch the dashboard, use the command appropriate for the framework implemented in `dashboard.py`.

For example, if the dashboard uses Streamlit:

```bash
streamlit run dashboard.py
```

## Model Evaluation

The project can evaluate predictive models using relevant classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The appropriate metrics and reported results depend on the implemented prediction target, test dataset, and evaluation procedure. Actual performance values should be documented only after running the experiments.

## Dataset Information

The project uses ICU-related datasets containing information such as admissions, ICU stays, patient records, laboratory events, chart events, and input/output events.

Document the original dataset source, version, permitted use, preprocessing steps, and prediction target here.

Ensure that the dataset is used in accordance with its licensing terms and privacy requirements. Do not publish personally identifiable or otherwise unauthorized patient information.

## Limitations

- Model performance depends on dataset quality, preprocessing, and the selected prediction target.
- Class imbalance and missing clinical information may affect predictions.
- Results from retrospective datasets may not generalize to other hospitals or patient populations.
- The system is an academic prototype and requires further validation before any clinical application.

## Future Enhancements

- Improve predictive performance through systematic model comparison.
- Add explainable AI visualizations to help interpret model outputs.
- Enhance the dashboard with interactive patient-data analysis.
- Evaluate generalization using independent datasets.
- Investigate integration with authorized healthcare information systems, subject to clinical, privacy, and regulatory requirements.

## Academic Disclaimer

This project was developed for educational and research purposes. It is not a certified medical device and must not be used independently to make patient discharge decisions.

## Author

Project: ICU Discharge Prediction Using Machine Learning  
Author: Kavya Pnadian 
Institution: SRM University 
Academic Year: 2026

## License

This project is distributed under the license specified in the `LICENSE` file. Review that file for the applicable terms.
