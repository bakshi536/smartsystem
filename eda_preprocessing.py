"""
Campus Analytics - EDA & Preprocessing
Run this BEFORE the dashboard or ML models.
Saves cleaned CSVs to campus_data/cleaned/
"""

import pandas as pd
import numpy as np
import os

DATA_DIR   = "campus_data"
CLEAN_DIR  = os.path.join(DATA_DIR, "cleaned")
os.makedirs(CLEAN_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading datasets...")
students   = pd.read_csv(f"{DATA_DIR}/students_master.csv")
attendance = pd.read_csv(f"{DATA_DIR}/attendance_logs.csv")
wifi       = pd.read_csv(f"{DATA_DIR}/wifi_usage.csv")
elec       = pd.read_csv(f"{DATA_DIR}/electricity_consumption.csv")
mess       = pd.read_csv(f"{DATA_DIR}/mess_footfall.csv")

# ── Attendance ────────────────────────────────────────────────────────────────
print("Cleaning attendance...")
attendance["date"]       = pd.to_datetime(attendance["date"])
attendance["entry_time"] = attendance["entry_time"].fillna("09:00")   # impute mode
attendance["exit_time"]  = attendance["exit_time"].fillna("17:00")
attendance["status_bin"] = (attendance["status"] == "Present").astype(int)
attendance["week"]       = attendance["date"].dt.isocalendar().week.astype(int)
attendance["month"]      = attendance["date"].dt.month

# Per-student attendance rate feature
att_rate = (attendance.groupby("student_id")["status_bin"]
            .mean().reset_index().rename(columns={"status_bin": "overall_att_rate"}))
attendance = attendance.merge(att_rate, on="student_id", how="left")

# 7-day rolling attendance (pivoted to student-day level first)
daily_att = (attendance.groupby(["student_id", "date"])["status_bin"]
             .mean().reset_index().rename(columns={"status_bin": "daily_att"}))
daily_att = daily_att.sort_values(["student_id", "date"])
daily_att["rolling_7d_att"] = (daily_att.groupby("student_id")["daily_att"]
                                .transform(lambda x: x.rolling(7, min_periods=1).mean()))
attendance = attendance.merge(daily_att[["student_id","date","rolling_7d_att"]],
                              on=["student_id","date"], how="left")

attendance.to_csv(f"{CLEAN_DIR}/attendance_clean.csv", index=False)
print(f"  ✓ Attendance cleaned: {len(attendance):,} rows")

# ── WiFi ──────────────────────────────────────────────────────────────────────
print("Cleaning WiFi...")
wifi["date"]         = pd.to_datetime(wifi["date"])
wifi["data_used_mb"] = wifi["data_used_mb"].fillna(wifi["data_used_mb"].median())
wifi["login_hour"]   = wifi["login_time"].str[:2].astype(int)
wifi["is_night"]     = wifi["login_hour"].between(21, 23).astype(int)

# Daily aggregates per student
wifi_daily = (wifi.groupby(["student_id","date"])
              .agg(total_mb=("data_used_mb","sum"),
                   sessions=("data_used_mb","count"),
                   avg_duration=("duration_minutes","mean"))
              .reset_index())
wifi_daily.to_csv(f"{CLEAN_DIR}/wifi_daily.csv", index=False)
wifi.to_csv(f"{CLEAN_DIR}/wifi_clean.csv", index=False)
print(f"  ✓ WiFi cleaned: {len(wifi):,} rows")

# ── Electricity ───────────────────────────────────────────────────────────────
print("Cleaning electricity...")
elec["date"]            = pd.to_datetime(elec["date"])
elec["meter_reading_kwh"] = elec["meter_reading_kwh"].fillna(
    elec.groupby("building")["meter_reading_kwh"].transform("median"))
elec["rolling_3d_avg"]  = (elec.groupby(["building","time_slot"])["consumption_kwh"]
                            .transform(lambda x: x.rolling(3, min_periods=1).mean()))
elec["anomaly_score"]   = (elec["consumption_kwh"] / elec["rolling_3d_avg"]).fillna(1)
elec.to_csv(f"{CLEAN_DIR}/electricity_clean.csv", index=False)
print(f"  ✓ Electricity cleaned: {len(elec):,} rows")

# ── Mess ──────────────────────────────────────────────────────────────────────
print("Cleaning mess...")
mess["date"] = pd.to_datetime(mess["date"], dayfirst=True)
mess["entry_time"] = mess["entry_time"].fillna("12:00")

# Daily footfall per meal
mess_daily = (mess.groupby(["date","meal_type","day_of_week","is_exam_week"])
              .size().reset_index(name="footfall"))
mess_daily["date"] = pd.to_datetime(mess_daily["date"])
mess_daily["prev_day_footfall"] = (mess_daily.groupby("meal_type")["footfall"]
                                   .shift(1).bfill())
mess_daily["rolling_7d_footfall"] = (mess_daily.groupby("meal_type")["footfall"]
                                     .transform(lambda x: x.rolling(7, min_periods=1).mean()))
mess_daily.to_csv(f"{CLEAN_DIR}/mess_daily.csv", index=False)
mess.to_csv(f"{CLEAN_DIR}/mess_clean.csv", index=False)
print(f"  ✓ Mess cleaned: {len(mess):,} rows")

# ── Master merged table (student-day level) ───────────────────────────────────
print("Building master merged table...")
master = daily_att.copy()
master = master.merge(wifi_daily, on=["student_id","date"], how="left")
master = master.merge(students[["student_id","department","section",
                                "semester","hostel_block"]], on="student_id", how="left")
master["total_mb"]    = master["total_mb"].fillna(0)
master["sessions"]    = master["sessions"].fillna(0)
master["day_of_week"] = master["date"].dt.day_name()
master["is_weekend"]  = master["date"].dt.dayofweek.isin([5,6]).astype(int)
master["week"]        = master["date"].dt.isocalendar().week.astype(int)
master["month"]       = master["date"].dt.month
master.to_csv(f"{CLEAN_DIR}/master_student_day.csv", index=False)
print(f"  ✓ Master table: {len(master):,} rows")

print("\n✅ All cleaned files saved to campus_data/cleaned/")
print("   Run the ML script or Streamlit dashboard next.")
