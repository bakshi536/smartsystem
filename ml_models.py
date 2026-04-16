"""
Campus Analytics - ML Models
Trains and saves all models to campus_data/models/
Run AFTER eda_preprocessing.py
"""

import pandas as pd
import numpy as np
import os, pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, f1_score, classification_report,
                                      mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing    import LabelEncoder, StandardScaler

CLEAN_DIR  = "campus_data/cleaned"
MODEL_DIR  = "campus_data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(obj, name):
    path = f"{MODEL_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"    Saved → {path}")

def load_csv(name):
    return pd.read_csv(f"{CLEAN_DIR}/{name}", parse_dates=["date"])

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — ATTENDANCE RISK PREDICTOR
# Goal: Will a student be absent tomorrow? (High / Medium / Low risk)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 1 — Attendance Risk Predictor")
print("="*60)

master = load_csv("master_student_day.csv")

# Features
le_dept = LabelEncoder()
le_day  = LabelEncoder()
master["dept_enc"] = le_dept.fit_transform(master["department"].fillna("Unknown"))
master["day_enc"]  = le_day.fit_transform(master["day_of_week"])

features = ["rolling_7d_att", "total_mb", "sessions", "dept_enc",
            "day_enc", "is_weekend", "week", "month"]

master_ml = master.dropna(subset=features + ["daily_att"])
X = master_ml[features]
y = pd.cut(master_ml["daily_att"],
           bins=[-0.01, 0.4, 0.7, 1.01],
           labels=["High Risk", "Medium Risk", "Low Risk"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

rf_att = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_att.fit(X_train, y_train)
y_pred = rf_att.predict(X_test)

print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  F1 Score : {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(classification_report(y_test, y_pred))

save_model(rf_att,  "attendance_risk_model")
save_model(le_dept, "label_enc_dept")
save_model(le_day,  "label_enc_day")

feature_imp_att = pd.Series(rf_att.feature_importances_, index=features).sort_values(ascending=False)
feature_imp_att.to_csv(f"{MODEL_DIR}/attendance_feature_importance.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — MESS FOOTFALL PREDICTOR
# Goal: How many students will come for each meal?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2 — Mess Footfall Predictor")
print("="*60)

mess_daily = load_csv("mess_daily.csv")
le_meal = LabelEncoder()
le_mday = LabelEncoder()
mess_daily["meal_enc"] = le_meal.fit_transform(mess_daily["meal_type"])
mess_daily["day_enc"]  = le_mday.fit_transform(mess_daily["day_of_week"])
mess_daily["month"]    = mess_daily["date"].dt.month
mess_daily["week"]     = mess_daily["date"].dt.isocalendar().week.astype(int)

feat_mess = ["meal_enc", "day_enc", "is_exam_week",
             "prev_day_footfall", "rolling_7d_footfall", "month", "week"]

mess_ml = mess_daily.dropna(subset=feat_mess + ["footfall"])
X_m = mess_ml[feat_mess]
y_m = mess_ml["footfall"]

X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

rf_mess = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_mess.fit(X_tr, y_tr)
y_pred_m = rf_mess.predict(X_te)

print(f"  MAE  : {mean_absolute_error(y_te, y_pred_m):.2f} students")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_te, y_pred_m)):.2f}")
print(f"  R²   : {r2_score(y_te, y_pred_m):.3f}")

save_model(rf_mess, "mess_footfall_model")
save_model(le_meal, "label_enc_meal")
save_model(le_mday, "label_enc_mess_day")

feature_imp_mess = pd.Series(rf_mess.feature_importances_, index=feat_mess).sort_values(ascending=False)
feature_imp_mess.to_csv(f"{MODEL_DIR}/mess_feature_importance.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — FOOD QUANTITY ESTIMATOR PER MEAL
# Goal: Estimate kg of food needed per meal (footfall × per-person avg)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 3 — Food Quantity Estimator")
print("="*60)

# Food consumption (grams per person) per meal — realistic college mess values
food_per_person = {
    "Breakfast": {"Rice/Upma": 200, "Dal/Curry": 150, "Bread": 100},
    "Lunch":     {"Rice": 350, "Dal": 200, "Sabzi": 150, "Roti": 120},
    "Snacks":    {"Tea": 150, "Snack Item": 100},
    "Dinner":    {"Rice": 300, "Dal": 200, "Sabzi": 150, "Roti": 120},
}

# Use predicted footfall from model 2 to estimate food
mess_pred = mess_daily.copy()
mess_pred["pred_footfall"] = rf_mess.predict(mess_daily[feat_mess].fillna(0))

food_rows = []
for meal, items in food_per_person.items():
    subset = mess_pred[mess_pred["meal_type"] == meal]
    for item, grams in items.items():
        subset = subset.copy()
        subset["food_item"]     = item
        subset["qty_kg_needed"] = (subset["pred_footfall"] * grams / 1000).round(2)
        food_rows.append(subset[["date","meal_type","food_item",
                                  "pred_footfall","qty_kg_needed"]])

food_df = pd.concat(food_rows).sort_values(["date","meal_type"])
food_df.to_csv(f"{MODEL_DIR}/food_quantity_estimates.csv", index=False)
print(f"  ✓ Food quantity estimates saved — {len(food_df):,} rows")
print(f"  Sample output:")
print(food_df[food_df["meal_type"]=="Lunch"].head(6).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — ELECTRICITY ANOMALY DETECTOR
# Goal: Flag unusual consumption (burst usage, equipment fault, etc.)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 4 — Electricity Anomaly Detector")
print("="*60)

elec = load_csv("electricity_clean.csv")
le_bld  = LabelEncoder()
le_slot = LabelEncoder()
elec["bld_enc"]  = le_bld.fit_transform(elec["building"])
elec["slot_enc"] = le_slot.fit_transform(elec["time_slot"])
elec["month"]    = elec["date"].dt.month
elec["dow"]      = elec["date"].dt.dayofweek

feat_elec = ["consumption_kwh", "rolling_3d_avg", "anomaly_score",
             "bld_enc", "slot_enc", "month", "dow"]
X_e = elec[feat_elec].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_e)

iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
elec["predicted_anomaly"] = iso.fit_predict(X_scaled)
elec["predicted_anomaly"] = (elec["predicted_anomaly"] == -1).astype(int)

# Evaluation vs labelled anomalies from simulation
tp = ((elec["predicted_anomaly"] == 1) & (elec["is_anomaly"] == True)).sum()
fp = ((elec["predicted_anomaly"] == 1) & (elec["is_anomaly"] == False)).sum()
fn = ((elec["predicted_anomaly"] == 0) & (elec["is_anomaly"] == True)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"  Precision : {precision:.3f}")
print(f"  Recall    : {recall:.3f}")
print(f"  Anomalies detected: {elec['predicted_anomaly'].sum()}")

elec.to_csv(f"{CLEAN_DIR}/electricity_with_anomalies.csv", index=False)
save_model(iso,    "electricity_anomaly_model")
save_model(scaler, "electricity_scaler")
save_model(le_bld, "label_enc_building")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — WiFi USAGE PREDICTOR
# Goal: Predict a student's WiFi usage tomorrow (MB)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 5 — WiFi Usage Predictor")
print("="*60)

wifi_daily = load_csv("wifi_daily.csv")
wifi_daily = wifi_daily.merge(
    pd.read_csv("campus_data/students_master.csv")[["student_id","department"]],
    on="student_id", how="left")
wifi_daily["dept_enc"] = le_dept.transform(
    wifi_daily["department"].fillna("Unknown").map(
        lambda x: x if x in le_dept.classes_ else le_dept.classes_[0]))
wifi_daily["dow"]   = wifi_daily["date"].dt.dayofweek
wifi_daily["week"]  = wifi_daily["date"].dt.isocalendar().week.astype(int)
wifi_daily["month"] = wifi_daily["date"].dt.month
wifi_daily["lag1_mb"] = wifi_daily.groupby("student_id")["total_mb"].shift(1)
wifi_daily["lag7_mb"] = wifi_daily.groupby("student_id")["total_mb"].shift(7)
wifi_daily = wifi_daily.fillna(0)

feat_wifi = ["lag1_mb","lag7_mb","sessions","dept_enc","dow","week","month"]
X_w = wifi_daily[feat_wifi]
y_w = wifi_daily["total_mb"]

X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
rf_wifi = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_wifi.fit(X_tr, y_tr)
y_pred_w = rf_wifi.predict(X_te)

print(f"  MAE  : {mean_absolute_error(y_te, y_pred_w):.2f} MB")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_te, y_pred_w)):.2f}")
print(f"  R²   : {r2_score(y_te, y_pred_w):.3f}")

save_model(rf_wifi, "wifi_usage_model")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ALL MODELS TRAINED SUCCESSFULLY")
print("="*60)
models = os.listdir(MODEL_DIR)
for m in sorted(models):
    print(f"  → {m}")
print("\nRun streamlit_app.py next for the dashboard!")
