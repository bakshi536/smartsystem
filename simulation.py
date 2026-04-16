"""
Campus Analytics - Data Simulation Script
Generates 4 datasets: Attendance, WiFi Usage, Electricity, Mess Footfall
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_STUDENTS   = 500
START_DATE     = datetime(2024, 1, 1)
END_DATE       = datetime(2024, 3, 31)  # ~90 days
DEPARTMENTS    = ["CSE", "ECE", "ME", "CE", "EE"]
SECTIONS       = ["A", "B", "C"]
SUBJECTS       = {
    "CSE": ["DBMS", "OS", "CN", "ML", "DSA"],
    "ECE": ["Signals", "VLSI", "Comm", "Microprocessors", "EMT"],
    "ME":  ["Thermodynamics", "FMM", "Manufacturing", "CAD", "MOM"],
    "CE":  ["Structures", "Geotechnics", "Hydraulics", "RCC", "Survey"],
    "EE":  ["Circuits", "Power Systems", "Machines", "Control", "PED"],
}
BUILDINGS      = ["Academic Block A", "Academic Block B", "Hostel Block 1",
                  "Hostel Block 2", "Library", "Labs Complex"]
OUTPUT_DIR     = "campus_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: date range ────────────────────────────────────────────────────────
all_dates = [START_DATE + timedelta(days=i)
             for i in range((END_DATE - START_DATE).days + 1)]
weekdays  = [d for d in all_dates if d.weekday() < 6]   # Mon–Sat (college)

# ── Exam weeks (lower attendance, higher electricity / WiFi) ──────────────────
exam_weeks = {datetime(2024, 2, 19) + timedelta(days=i) for i in range(7)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STUDENT MASTER TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating student master table...")
students = []
for sid in range(1, NUM_STUDENTS + 1):
    dept = random.choice(DEPARTMENTS)
    students.append({
        "student_id":  f"STU{sid:04d}",
        "name":        f"Student_{sid}",
        "department":  dept,
        "section":     random.choice(SECTIONS),
        "semester":    random.choice([1, 3, 5, 7]),
        "hostel_block": random.choice(["H1", "H2", "H3", "Day Scholar"]),
        "year_of_study": random.choice([1, 2, 3, 4]),
    })
students_df = pd.DataFrame(students)
student_ids = students_df["student_id"].tolist()
dept_map    = dict(zip(students_df["student_id"], students_df["department"]))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ATTENDANCE LOGS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating attendance logs...")

attendance_records = []
for sid in student_ids:
    dept = dept_map[sid]
    subjects = SUBJECTS[dept]
    # Each student has a base attendance probability (some are naturally irregular)
    base_prob = np.random.beta(8, 2)  # most students attend ~80%

    for date in weekdays:
        day_name = date.strftime("%A")
        is_exam  = date in exam_weeks

        # Adjust probability based on day & exam
        prob = base_prob
        if day_name == "Monday":   prob -= 0.08   # Monday blues
        if day_name == "Saturday": prob -= 0.05
        if is_exam:                prob -= 0.15   # skip lectures near exams
        prob = np.clip(prob, 0.1, 1.0)

        for subject in subjects:
            absent_prob = max(0, 1 - prob - 0.03)
            late_prob   = 0.03
            present_prob = 1 - absent_prob - late_prob
            if present_prob <= 0:
                status = np.random.choice(["Present", "Absent"], p=[prob, 1 - prob])
            else:
                status = np.random.choice(
                    ["Present", "Absent", "Late"],
                    p=[present_prob, absent_prob, late_prob]
                )

            entry_time = None
            exit_time  = None
            if status == "Present":
                entry_h = random.randint(8, 9)
                entry_m = random.randint(0, 20)
                entry_time = f"{entry_h:02d}:{entry_m:02d}"
                exit_h  = random.randint(16, 18)
                exit_time  = f"{exit_h:02d}:{random.randint(0,59):02d}"
            elif status == "Late":
                entry_time = f"09:{random.randint(20,59):02d}"
                exit_time  = f"{random.randint(16,17):02d}:{random.randint(0,59):02d}"

            # Introduce ~3% missing values in entry_time
            if random.random() < 0.03:
                entry_time = np.nan

            attendance_records.append({
                "student_id":  sid,
                "date":        date.strftime("%Y-%m-%d"),
                "day_of_week": day_name,
                "subject":     subject,
                "department":  dept,
                "status":      status,
                "entry_time":  entry_time,
                "exit_time":   exit_time,
                "is_exam_week": is_exam,
            })

attendance_df = pd.DataFrame(attendance_records)
attendance_df.to_csv(f"{OUTPUT_DIR}/attendance_logs.csv", index=False)
print(f"  ✓ Attendance: {len(attendance_df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WIFI USAGE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating WiFi usage data...")

access_points = ["Library", "Block_A_F1", "Block_A_F2", "Block_B_F1",
                 "Hostel_H1", "Hostel_H2", "Canteen", "Labs"]

wifi_records = []
for sid in student_ids:
    is_heavy_user = random.random() < 0.15   # 15% heavy users

    for date in all_dates:
        day_name = date.strftime("%A")
        is_exam  = date in exam_weeks

        # Some students don't connect every day
        if random.random() < 0.1:   # 10% chance of no WiFi that day
            continue

        # Number of sessions per day
        sessions = random.randint(1, 4)
        for _ in range(sessions):
            # Peak usage: 9PM–12AM (evening in hostel), 10AM–1PM (daytime)
            hour_weights = [1]*7 + [3]*4 + [2]*3 + [1]*3 + [4]*3 + [1]*4  # 24h
            login_hour = random.choices(range(24), weights=hour_weights)[0]
            login_min  = random.randint(0, 59)
            duration   = int(np.random.lognormal(3.5, 0.8))   # minutes, skewed
            duration   = np.clip(duration, 5, 300)
            if is_heavy_user: duration = int(duration * 1.5)
            if is_exam:       duration = int(duration * 1.3)

            data_mb = round(duration * np.random.uniform(0.5, 3.5), 2)
            if is_heavy_user: data_mb *= 2

            # ~2% missing data_used (packet loss / logging error)
            if random.random() < 0.02:
                data_mb = np.nan

            wifi_records.append({
                "student_id":          sid,
                "date":                date.strftime("%Y-%m-%d"),
                "day_of_week":         day_name,
                "login_time":          f"{login_hour:02d}:{login_min:02d}",
                "duration_minutes":    duration,
                "data_used_mb":        data_mb,
                "access_point":        random.choice(access_points),
                "device_type":         random.choice(["Mobile", "Laptop", "Tablet"]),
                "is_exam_week":        is_exam,
            })

wifi_df = pd.DataFrame(wifi_records)
wifi_df.to_csv(f"{OUTPUT_DIR}/wifi_usage.csv", index=False)
print(f"  ✓ WiFi: {len(wifi_df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ELECTRICITY CONSUMPTION RECORDS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating electricity consumption records...")

time_slots = ["06-08", "08-10", "10-12", "12-14",
              "14-16", "16-18", "18-20", "20-22", "22-00"]

# Base kWh per slot per building (realistic profile)
slot_base = {
    "06-08": 0.6, "08-10": 1.8, "10-12": 2.2, "12-14": 1.9,
    "14-16": 2.0, "16-18": 1.7, "18-20": 1.5, "20-22": 1.2, "22-00": 0.5,
}

elec_records = []
record_id = 1
prev_reading = {b: round(random.uniform(1000, 5000), 2) for b in BUILDINGS}

for date in all_dates:
    day_type = ("weekend" if date.weekday() == 6 else
                "saturday" if date.weekday() == 5 else "weekday")
    is_exam  = date in exam_weeks

    for building in BUILDINGS:
        for slot, base_kwh in slot_base.items():
            # Scale by building size
            scale = 1.5 if "Academic" in building else (
                    0.8 if "Hostel" in building else 1.0)
            if day_type == "weekend": scale *= 0.4
            if day_type == "saturday": scale *= 0.7
            if is_exam: scale *= 1.2

            consumption = round(base_kwh * scale * np.random.uniform(0.85, 1.15), 3)

            # Inject anomalies ~1% of the time
            is_anomaly = False
            if random.random() < 0.01:
                consumption *= random.uniform(2.5, 4.0)
                is_anomaly = True

            # ~2% missing meter readings
            meter_reading = round(prev_reading[building] + consumption, 2)
            if random.random() < 0.02:
                meter_reading = np.nan

            prev_reading[building] = meter_reading if not np.isnan(meter_reading) \
                                     else prev_reading[building] + consumption

            elec_records.append({
                "record_id":        f"ELEC{record_id:06d}",
                "date":             date.strftime("%Y-%m-%d"),
                "day_type":         day_type,
                "building":         building,
                "time_slot":        slot,
                "consumption_kwh":  consumption,
                "meter_reading_kwh": meter_reading,
                "is_exam_week":     is_exam,
                "is_anomaly":       is_anomaly,
                "month":            date.strftime("%B"),
            })
            record_id += 1

elec_df = pd.DataFrame(elec_records)
elec_df.to_csv(f"{OUTPUT_DIR}/electricity_consumption.csv", index=False)
print(f"  ✓ Electricity: {len(elec_df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MESS ENTRY / FOOTFALL DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating mess footfall data...")

meals = {
    "Breakfast": ("07:00", "09:00", 0.35),   # (window_start, end, base_rate)
    "Lunch":     ("12:00", "14:00", 0.75),
    "Snacks":    ("16:30", "17:30", 0.45),
    "Dinner":    ("19:00", "21:00", 0.82),
}

mess_records = []
entry_id = 1

for date in all_dates:
    day_name  = date.strftime("%A")
    is_sunday = date.weekday() == 6
    is_exam   = date in exam_weeks

    for meal, (t_start, t_end, base_rate) in meals.items():
        rate = base_rate
        if is_sunday and meal == "Breakfast": rate -= 0.15  # lazy Sundays
        if is_exam:                           rate += 0.05  # eat before studying
        rate = np.clip(rate, 0, 1)

        # Decide which students come
        attendees = [s for s in student_ids if random.random() < rate]

        # Spread arrival times uniformly within the meal window
        h_start, m_start = map(int, t_start.split(":"))
        h_end,   m_end   = map(int, t_end.split(":"))
        window_minutes = (h_end * 60 + m_end) - (h_start * 60 + m_start)

        for sid in attendees:
            offset = random.randint(0, window_minutes)
            arrival_min = h_start * 60 + m_start + offset
            entry_time  = f"{arrival_min // 60:02d}:{arrival_min % 60:02d}"

            # ~1.5% missing entry times (scan failure)
            if random.random() < 0.015:
                entry_time = np.nan

            mess_records.append({
                "entry_id":    f"MESS{entry_id:07d}",
                "student_id":  sid,
                "date":        date.strftime("%Y-%m-%d"),
                "day_of_week": day_name,
                "meal_type":   meal,
                "entry_time":  entry_time,
                "is_sunday":   is_sunday,
                "is_exam_week": is_exam,
                "mess_block":  random.choice(["Mess_A", "Mess_B"]),
            })
            entry_id += 1

mess_df = pd.DataFrame(mess_records)
mess_df.to_csv(f"{OUTPUT_DIR}/mess_footfall.csv", index=False)
print(f"  ✓ Mess: {len(mess_df):,} rows")

# ── Save student master ───────────────────────────────────────────────────────
students_df.to_csv(f"{OUTPUT_DIR}/students_master.csv", index=False)
print(f"  ✓ Students master: {len(students_df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  DATA SIMULATION COMPLETE")
print("="*55)
datasets = {
    "students_master.csv":       students_df,
    "attendance_logs.csv":       attendance_df,
    "wifi_usage.csv":            wifi_df,
    "electricity_consumption.csv": elec_df,
    "mess_footfall.csv":         mess_df,
}
for fname, df in datasets.items():
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    print(f"  {fname:<35} {len(df):>8,} rows  |  {null_pct}% nulls")

print(f"\n  Files saved to: ./{OUTPUT_DIR}/")
print("="*55)