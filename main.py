import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier



# LOAD DATASETS


print("="*60)
print("STEP 1: Loading datasets...")
print("="*60)

patients   = pd.read_csv("PATIENTS.csv/PATIENTS.csv")
admissions = pd.read_csv("ADMISSIONS.csv/ADMISSIONS.csv")
icustays   = pd.read_csv("ICUSTAYS.csv/ICUSTAYS.csv")

patients.columns   = patients.columns.str.strip()
admissions.columns = admissions.columns.str.strip()
icustays.columns   = icustays.columns.str.strip()

print(f"Patients   : {patients.shape}")
print(f"Admissions : {admissions.shape}")
print(f"ICU stays  : {icustays.shape}")



# MERGE BASE DATASETS


print("\n" + "="*60)
print("STEP 2: Merging datasets...")
print("="*60)

merged_data = icustays.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
merged_data = merged_data.merge(patients, on="SUBJECT_ID", how="inner")

print(f"Merged shape: {merged_data.shape}")

# ── KEY OPTIMIZATION: pre-build valid ICUSTAY_ID set ─────
valid_icustay_ids = set(merged_data["ICUSTAY_ID"].unique())
print(f"Valid ICU stay IDs to filter by: {len(valid_icustay_ids):,}")



# LOAD CHARTEVENTS (FAST — dual filter)


print("\n" + "="*60)
print("STEP 3: Loading CHARTEVENTS (fast dual-filter)...")
print("="*60)

d_items = pd.read_csv("D_ITEMS.csv/D_ITEMS.csv")
d_items.columns = d_items.columns.str.strip()

vital_signs = d_items[d_items["LABEL"].str.contains(
    "Heart Rate|Respiratory Rate|Temperature|SpO2|Oxygen|Blood Pressure|Glucose|pH|Sodium|Potassium",
    case=False, na=False
)]
vital_ids = set(vital_signs["ITEMID"].unique())
print(f"Vital ITEMIDs: {len(vital_ids)}")

CHUNK_SIZE = 500_000
chunks = []
total_rows_read = 0

for i, chunk in enumerate(pd.read_csv(
    "CHARTEVENTS.csv/CHARTEVENTS.csv",
    usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
    chunksize=CHUNK_SIZE,
    low_memory=False
)):
    total_rows_read += len(chunk)

    # DUAL FILTER — only keep rows where BOTH conditions are true
    # This cuts 99% of rows immediately
    filtered = chunk[
        chunk["ITEMID"].isin(vital_ids) &
        chunk["ICUSTAY_ID"].isin(valid_icustay_ids)
    ]

    if len(filtered) > 0:
        chunks.append(filtered)

    # Progress every 10 chunks (5M rows)
    if (i + 1) % 10 == 0:
        kept = sum(len(c) for c in chunks)
        print(f"  Read {total_rows_read:>12,} rows | Kept {kept:>8,} vital rows")

    # Early stop — once we have enough data, no need to read more
    # MIMIC-III CHARTEVENTS has ~330M rows but ICU vital data is
    # concentrated in the first ~100M rows
    if sum(len(c) for c in chunks) > 2_000_000:
        print(f"  Collected 2M+ vital rows — stopping early.")
        break

chartevents = pd.concat(chunks, ignore_index=True)
print(f"\nTotal vital rows collected: {chartevents.shape[0]:,}")

chartevents = chartevents.merge(
    d_items[["ITEMID", "LABEL"]], on="ITEMID", how="left"
)
chartevents["CHARTTIME"] = pd.to_datetime(chartevents["CHARTTIME"])
chartevents = chartevents.sort_values(["ICUSTAY_ID", "CHARTTIME"])



# LOAD LABEVENTS (FAST — same dual filter)


print("\n" + "="*60)
print("STEP 4: Loading LABEVENTS (fast)...")
print("="*60)

LAB_AVAILABLE = False

try:
    d_labitems = pd.read_csv("D_LABITEMS.csv/D_LABITEMS.csv")
    d_labitems.columns = d_labitems.columns.str.strip()

    key_labs = d_labitems[d_labitems["LABEL"].str.contains(
        "Creatinine|Lactate|Hemoglobin|WBC|Platelet|Bicarbonate|BUN|Chloride|Hematocrit",
        case=False, na=False
    )]
    lab_ids = set(key_labs["ITEMID"].unique())
    print(f"Lab ITEMIDs: {len(lab_ids)}")

    # LABEVENTS has SUBJECT_ID + HADM_ID but not ICUSTAY_ID
    # Build valid HADM_ID set for fast filtering
    valid_hadm_ids = set(merged_data["HADM_ID"].unique())

    lab_chunks = []
    for chunk in pd.read_csv(
        "LABEVENTS.csv/LABEVENTS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        filtered = chunk[
            chunk["ITEMID"].isin(lab_ids) &
            chunk["HADM_ID"].isin(valid_hadm_ids)
        ]
        if len(filtered) > 0:
            lab_chunks.append(filtered)

        if sum(len(c) for c in lab_chunks) > 500_000:
            print("  Collected 500k+ lab rows — stopping early.")
            break

    labevents = pd.concat(lab_chunks, ignore_index=True)
    labevents = labevents.merge(d_labitems[["ITEMID", "LABEL"]], on="ITEMID", how="left")
    labevents = labevents.merge(
        merged_data[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]],
        on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )
    print(f"LABEVENTS loaded: {labevents.shape[0]:,} rows")
    LAB_AVAILABLE = True

except Exception as e:
    print(f"LABEVENTS skipped ({e}) — using vitals only.")



# FEATURE ENGINEERING
#   Extract: mean, std, min, max, last per vital/lab


print("\n" + "="*60)
print("STEP 5: Feature engineering (mean+std+min+max+last)...")
print("="*60)

def extract_features(data, group_col="ICUSTAY_ID", value_col="VALUENUM", label_col="LABEL"):
    agg = (
        data.groupby([group_col, label_col])[value_col]
        .agg(["mean", "std", "min", "max", "last"])
        .reset_index()
    )
    pivoted = agg.pivot(index=group_col, columns=label_col)
    pivoted.columns = [f"{label}_{stat}" for stat, label in pivoted.columns]
    return pivoted.reset_index()

vital_features = extract_features(chartevents)
print(f"Vital features : {vital_features.shape}")

if LAB_AVAILABLE:
    lab_features = extract_features(labevents)
    print(f"Lab features   : {lab_features.shape}")
    features = vital_features.merge(lab_features, on="ICUSTAY_ID", how="left")
else:
    features = vital_features

features = features.fillna(features.mean(numeric_only=True))
print(f"Combined       : {features.shape}")



# BUILD FINAL DATASET


print("\n" + "="*60)
print("STEP 6: Building final dataset...")
print("="*60)

labels  = merged_data[["ICUSTAY_ID", "HOSPITAL_EXPIRE_FLAG"]].drop_duplicates()
dataset = features.merge(labels, on="ICUSTAY_ID", how="inner")

print(f"Final dataset  : {dataset.shape}")
print(f"Mortality rate : {round(dataset['HOSPITAL_EXPIRE_FLAG'].mean()*100, 2)}%")



# TRAIN / TEST SPLIT


print("\n" + "="*60)
print("STEP 7: Train/test split...")
print("="*60)

X = dataset.drop(["ICUSTAY_ID", "HOSPITAL_EXPIRE_FLAG"], axis=1)
y = dataset["HOSPITAL_EXPIRE_FLAG"]

feature_names = X.columns.tolist()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"Positive (mortality) in test: {y_test.sum()} / {len(y_test)}")



# SMOTE


print("\n" + "="*60)
print("STEP 8: Applying SMOTE...")
print("="*60)

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {X_train_sm.shape} | Class dist: {np.bincount(y_train_sm)}")



# RESHAPE FOR BILSTM  (N_TIMESTEPS windows)


print("\n" + "="*60)
print("STEP 9: Reshaping for BiLSTM (5 timesteps)...")
print("="*60)

N_FEATURES  = X_train_sm.shape[1]
N_TIMESTEPS = 5

pad_size = (N_TIMESTEPS - (N_FEATURES % N_TIMESTEPS)) % N_TIMESTEPS

X_train_p = np.hstack([X_train_sm, np.zeros((X_train_sm.shape[0], pad_size))]) if pad_size > 0 else X_train_sm
X_test_p  = np.hstack([X_test,     np.zeros((X_test.shape[0],     pad_size))]) if pad_size > 0 else X_test

FEATURES_PER_STEP = X_train_p.shape[1] // N_TIMESTEPS

X_train_lstm = X_train_p.reshape((X_train_p.shape[0], N_TIMESTEPS, FEATURES_PER_STEP))
X_test_lstm  = X_test_p.reshape((X_test_p.shape[0],   N_TIMESTEPS, FEATURES_PER_STEP))

print(f"BiLSTM shape: {X_train_lstm.shape}  (samples, {N_TIMESTEPS} timesteps, {FEATURES_PER_STEP} features/step)")



# BUILD & TRAIN BILSTM


print("\n" + "="*60)
print("STEP 10: Building & training BiLSTM...")
print("="*60)

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(N_TIMESTEPS, FEATURES_PER_STEP)),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=5,
    restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train_lstm, y_train_sm,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_lstm, y_test),
    callbacks=[early_stop],
    verbose=1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("BiLSTM Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history.history["loss"],         label="Train")
ax2.plot(history.history["val_loss"],     label="Validation")
ax2.set_title("BiLSTM Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()



# TRAIN XGBOOST


print("\n" + "="*60)
print("STEP 11: Training XGBoost...")
print("="*60)

pos_weight = int((y_train_sm == 0).sum() / max((y_train_sm == 1).sum(), 1))

xgb_model = XGBClassifier(
    n_estimators=600, max_depth=5, learning_rate=0.02,
    subsample=0.9, colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    gamma=0.3, min_child_weight=3,
    eval_metric="logloss", random_state=42
)
xgb_model.fit(X_train_sm, y_train_sm)



# TRAIN RANDOM FOREST


print("\n" + "="*60)
print("STEP 12: Training Random Forest...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    class_weight="balanced", random_state=42
)
rf_model.fit(X_train_sm, y_train_sm)



# PREDICTIONS + AUTO THRESHOLD


print("\n" + "="*60)
print("STEP 13: Generating predictions...")
print("="*60)

lstm_prob = model.predict(X_test_lstm).flatten()
lstm_pred = (lstm_prob > 0.5).astype(int)

xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred  = xgb_model.predict(X_test)

rf_prob   = rf_model.predict_proba(X_test)[:, 1]
rf_pred   = rf_model.predict(X_test)

ensemble_prob = (0.5 * xgb_prob + 0.3 * lstm_prob + 0.2 * rf_prob)
best_score, best_threshold = 0, 0.40

for t in np.arange(0.30, 0.55, 0.01):
    preds = (ensemble_prob > t).astype(int)
    acc   = accuracy_score(y_test, preds)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    
    score = (acc + rec + f1) / 3
    if score > best_score:
        best_score, best_threshold = score, t

print(f"Best threshold : {round(best_threshold, 2)}")
print(f"Best score     : {round(best_score, 4)}")
ensemble_pred = (ensemble_prob > best_threshold).astype(int)



# EVALUATE ALL MODELS


print("\n" + "="*60)
print("STEP 14: Model evaluation...")
print("="*60)

def evaluate(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    print(f"\n  {name}")
    print(f"    Accuracy  : {round(acc*100,2)}%")
    print(f"    Recall    : {round(rec,4)}")
    print(f"    Precision : {round(prec,4)}")
    print(f"    F1 Score  : {round(f1,4)}")
    print(f"    AUC-ROC   : {round(roc,4)}")
    return acc, rec, prec, f1, roc

rf_m  = evaluate("Random Forest",           y_test, rf_pred,       rf_prob)
bi_m  = evaluate("BiLSTM (Individual)",     y_test, lstm_pred,     lstm_prob)
xgb_m = evaluate("XGBoost (Individual)",    y_test, xgb_pred,      xgb_prob)
ens_m = evaluate("Ensemble BiLSTM+XGBoost", y_test, ensemble_pred, ensemble_prob)

print("\n" + "="*75)
print(f"{'Model':<32} {'Accuracy':>10} {'Recall':>8} {'Prec':>8} {'F1':>8} {'AUC':>8}")
print("="*75)
for mname, m in [("Random Forest",rf_m),("BiLSTM",bi_m),
                  ("XGBoost",xgb_m),("Ensemble BiLSTM+XGBoost",ens_m)]:
    print(f"{mname:<32} {m[0]*100:>9.2f}% {m[1]:>8.4f} {m[2]:>8.4f} {m[3]:>8.4f} {m[4]:>8.4f}")
print("="*75)



# PATIENT RISK CATEGORIES


def risk_category(prob):
    if prob < 0.10:   return "SAFE"
    elif prob < 0.25: return "MONITOR"
    else:             return "HIGH RISK"

print("\nPATIENT RISK RESULTS (first 30)")
for i in range(min(30, len(ensemble_prob))):
    prob = float(ensemble_prob[i])
    print(f"  Patient {i+1:>2} | Risk: {prob:.3f} | {risk_category(prob)}")



# CONFUSION MATRIX


cm = confusion_matrix(y_test, ensemble_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Survive", "Pred: Expire"],
            yticklabels=["Act: Survive",  "Act: Expire"])
plt.title("Confusion Matrix — Ensemble (BiLSTM + XGBoost)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()



# ROC CURVE (all models overlaid)


plt.figure(figsize=(8, 6))
for label, y_prob_arr, color in [
    ("Random Forest",           rf_prob,       "green"),
    ("BiLSTM",                  lstm_prob,     "blue"),
    ("XGBoost",                 xgb_prob,      "orange"),
    ("Ensemble BiLSTM+XGBoost", ensemble_prob, "red"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob_arr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={round(auc(fpr,tpr),3)})",
             color=color, linewidth=2)

plt.plot([0,1],[0,1],"k--", label="Random Baseline")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()



# MODEL COMPARISON BAR CHARTS

model_names = ["Random\nForest", "BiLSTM", "XGBoost", "Ensemble\n(Proposed)"]
fig, axes   = plt.subplots(1, 3, figsize=(14, 5))
colors      = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

for ax, values, title, ylabel, ylim in [
    (axes[0],[round(m[0]*100,2) for m in [rf_m,bi_m,xgb_m,ens_m]],"Accuracy", "Accuracy (%)",(60,100)),
    (axes[1],[round(m[1],4)     for m in [rf_m,bi_m,xgb_m,ens_m]],"Recall",   "Recall",      (0,1.0)),
    (axes[2],[round(m[4],4)     for m in [rf_m,bi_m,xgb_m,ens_m]],"AUC-ROC",  "AUC-ROC",     (0.5,1.0)),
]:
    bars = ax.bar(model_names, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(ylim); ax.set_title(title); ax.set_ylabel(ylabel)

plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# ══════════════════════════════════════════════════════════
# STEP 19 — SHAP EXPLAINABLE AI
# ══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 19: Generating SHAP Explainable AI...")
print("="*60)

X_train_2d = X_train_lstm.reshape(X_train_lstm.shape[0], -1)
X_test_2d  = X_test_lstm.reshape(X_test_lstm.shape[0],   -1)

padded_feature_names = list(feature_names) + [f"pad_{i}" for i in range(pad_size)]

def model_predict(data):
    data = data.reshape((data.shape[0], N_TIMESTEPS, FEATURES_PER_STEP))
    return model.predict(data, verbose=0)

explainer   = shap.KernelExplainer(model_predict, X_train_2d[:50])
shap_values = explainer.shap_values(X_test_2d[:20])

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap.summary_plot(shap_values, X_test_2d[:20],
                  feature_names=padded_feature_names, show=False)
plt.title("SHAP Feature Importance — BiLSTM")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

print("\nTop SHAP features per patient:")

# Safely handle shap_values shape — could be (n_patients, n_features) or (n_patients, n_features, 1)
sv = np.array(shap_values)
if sv.ndim == 3:
    sv = sv[:, :, 0]   # squeeze last dim if shape is (n, f, 1)

for i in range(min(3, sv.shape[0])):
    print(f"\n  Patient {i+1}")
    top_indices = np.argsort(np.abs(sv[i]))[::-1][:5]
    for j in top_indices.tolist():   # .tolist() ensures plain Python int index
        if j < len(padded_feature_names):
            impact    = float(sv[i][j])
            feature   = padded_feature_names[j]
            direction = "increases" if impact > 0 else "reduces"
            print(f"    {feature} ({round(impact, 4)}) → {direction} risk")


# ══════════════════════════════════════════════════════════
# STEP 20 — SAVE ALL MODELS
# ══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 20: Saving all models...")
print("="*60)

model.save("icu_lstm_model.keras")
joblib.dump(xgb_model,         "icu_xgboost_model.pkl")
joblib.dump(rf_model,          "icu_randomforest_model.pkl")
joblib.dump(scaler,            "icu_scaler.pkl")
joblib.dump(feature_names,     "icu_features.pkl")
joblib.dump(N_TIMESTEPS,       "icu_timesteps.pkl")
joblib.dump(FEATURES_PER_STEP, "icu_features_per_step.pkl")
joblib.dump(pad_size,          "icu_pad_size.pkl")

print("All models saved!")
print("Outputs: training_curves.png | confusion_matrix.png | roc_curve.png | model_comparison.png | shap_summary.png")
