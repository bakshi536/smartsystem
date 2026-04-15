import pandas as pd
import random
from datetime import datetime, timedelta

NUM_ROWS = 10000

student_ids = [f"S{1000+i}" for i in range(NUM_ROWS)]
bhawans = ["Rajendra", "Sarojini", "Azad", "Ganga", "Yamuna"]

start_date = datetime(2026, 4, 1)

# ---------------------------
# TIME FORMAT
# ---------------------------
def format_time(hour, minute):
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12
    h = 12 if h == 0 else h
    return f"{h:02d}:{minute:02d} {suffix}"

def to_minutes(h, m):
    return h * 60 + m

# ---------------------------
# EXAM SEASON
# ---------------------------
def is_exam_season(date):
    return 1 if date.day >= 20 else 0

# ---------------------------
# MESS
# ---------------------------
def generate_mess():
    slot = random.choice(["Breakfast", "Lunch", "Dinner"])

    if slot == "Breakfast":
        h1, h2 = 7, 9
    elif slot == "Lunch":
        h1, h2 = 12, 14
    else:
        h1, h2 = 19, 21

    h = random.randint(h1, h2)
    m = random.randint(0, 59)

    start = to_minutes(h, m)
    end = start + random.randint(20, 60)
    curr = random.randint(start, end)

    return start, end, curr, slot

# ---------------------------
# LIBRARY
# ---------------------------
def generate_library(exam):
    h = random.randint(9, 22)
    m = random.randint(0, 59)

    start = to_minutes(h, m)

    if exam:
        duration = random.randint(300, 540)
    else:
        duration = random.randint(30, 180)

    end = min(start + duration, 23 * 60 + 59)
    curr = random.randint(start, end)

    return start, end, curr

# ---------------------------
# RANDOM TIME
# ---------------------------
def generate_random():
    return random.randint(0, 23 * 60 + 59)

# ---------------------------
# WIFI + ELECTRICITY
# ---------------------------
def wifi_usage(hour, loc):
    base = random.uniform(100, 400)
    if loc == "Hostel" and hour >= 20:
        return base + random.uniform(500, 1200)
    if loc == "Library":
        return base + random.uniform(300, 700)
    if loc == "Mess":
        return base + random.uniform(100, 300)
    return base

def electricity_usage(hour, loc, users):
    base = users * random.uniform(0.05, 0.15)
    if loc == "Mess":
        return base + random.uniform(20, 40)
    if loc == "Hostel":
        return base + random.uniform(10, 25)
    if loc == "Library":
        return base + random.uniform(5, 15)
    return base

# ---------------------------
# DATA GENERATION
# ---------------------------
data = []

for i in range(NUM_ROWS):

    sid = student_ids[i]
    bhawan = random.choice(bhawans)
    date = start_date + timedelta(days=random.randint(0, 30))

    exam = is_exam_season(date)

    activity = random.choice(["Mess", "Library", "Other"])

    if activity == "Mess":
        start, end, curr, meal = generate_mess()
        location = "Mess"
        m_entry, m_exit = start, end
        l_entry, l_exit = 0, 0
        food_category = meal

    elif activity == "Library":
        start, end, curr = generate_library(exam)
        location = "Library"
        l_entry, l_exit = start, end
        m_entry, m_exit = 0, 0
        food_category = "None"

    else:
        curr = generate_random()
        hour = curr // 60

        if 8 <= hour <= 18:
            location = random.choice(["Classroom", "Hostel", "Outside"])
        else:
            location = random.choice(["Hostel", "Outside"])

        m_entry, m_exit = 0, 0
        l_entry, l_exit = 0, 0
        food_category = "None"

    hour = curr // 60
    minute = curr % 60

    # ---------------------------
    # HOSTEL MOVEMENT
    # ---------------------------
    if location in ["Hostel", "Mess"]:
        leave = 0
        back = 0

    elif location == "Library":
        leave = l_entry - random.randint(5, 30)
        back = l_exit + random.randint(10, 60)

    elif location == "Classroom":
        leave = curr - random.randint(10, 40)
        back = curr + random.randint(30, 120)

    else:
        leave = curr - random.randint(30, 120)
        back = curr + random.randint(60, 180)

    leave = max(0, leave)
    back = min(23 * 60 + 59, back)

    # ---------------------------
    # SLEEP HOURS (INTEGER)
    # ---------------------------
    r = random.random()

    if exam:
        if r < 0.5:
            sleep = random.uniform(3, 5)
        elif r < 0.8:
            sleep = random.uniform(5, 6.5)
        else:
            sleep = random.uniform(6.5, 8)
    else:
        if r < 0.05:
            sleep = random.uniform(10, 17)
        elif r < 0.2:
            sleep = random.uniform(8, 10)
        elif r < 0.85:
            sleep = random.uniform(6, 8)
        else:
            sleep = random.uniform(3, 6)

    sleep = int(round(sleep))  # ✅ INTEGER

    # ---------------------------
    # FORMAT FUNCTION
    # ---------------------------
    def conv(x):
        if x == 0:
            return "0"
        return format_time(x // 60, x % 60)

    time_str = format_time(hour, minute)

    # ---------------------------
    # USERS + ENERGY
    # ---------------------------
    users = random.randint(10, 200)
    wifi = round(wifi_usage(hour, location), 2)
    elec = round(electricity_usage(hour, location, users), 2)

    # ---------------------------
    # ATTENDANCE
    # ---------------------------
    if location == "Classroom":
        attendance = 1
    elif 9 <= hour <= 16:
        attendance = random.choice([0, 1])
    else:
        attendance = 0

    auto = 1 if (8 <= hour <= 10 or 16 <= hour <= 18) and attendance == 1 else 0

    data.append([
        sid, date.strftime("%Y-%m-%d"), time_str,
        bhawan, location,
        conv(m_entry), conv(m_exit),
        conv(l_entry), conv(l_exit),
        conv(leave), conv(back),
        wifi, users, elec,
        food_category, attendance, auto,
        exam, sleep
    ])

columns = [
    "Student_ID","Date","Time","Bhawan","Location",
    "Mess_Entry_Time","Mess_Exit_Time",
    "Library_Entry_Time","Library_Exit_Time",
    "Leaves_Hostel","Back_Hostel",
    "WiFi_MB","Active_Users","Electricity_Usage",
    "Food_Category","Attendance","Auto_Taken",
    "Exam_Season","Sleep_Hours"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("11smart_campus_final_clean.csv", index=False)

print("\n📊 FORMATTED TABLE:\n")
print(df.head(15).to_string(index=False))