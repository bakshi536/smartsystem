"""
Campus Analytics — Complete Single Page Dashboard
Run: /c/Python313/python.exe -m streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Campus Analytics", page_icon="🎓",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.main, .block-container { background: #070b14 !important; padding: 1rem 2rem !important; }
section[data-testid="stSidebar"] { display: none; }
.hero { background: linear-gradient(135deg,#0f172a,#1e1b4b 50%,#0f172a); border:1px solid #312e81; border-radius:16px; padding:2rem 2.5rem; margin-bottom:1.5rem; }
.hero h1 { font-size:2.4rem !important; font-weight:800 !important; color:#e2e8f0 !important; margin:0 0 0.3rem 0 !important; }
.hero p { color:#94a3b8; margin:0; font-size:1rem; }
.section-title { font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.15em; color:#6366f1; margin:2rem 0 1rem 0; display:flex; align-items:center; gap:0.5rem; }
.section-title::after { content:''; flex:1; height:1px; background:linear-gradient(to right,#312e81,transparent); }
.kpi-card { background:linear-gradient(145deg,#0f172a,#1e293b); border:1px solid #1e293b; border-radius:12px; padding:1.2rem 1.4rem; position:relative; overflow:hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; background:var(--accent); border-radius:3px 0 0 3px; }
.kpi-icon { font-size:1.5rem; margin-bottom:0.5rem; }
.kpi-val { font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:700; color:var(--accent); line-height:1; }
.kpi-label { font-size:0.78rem; color:#64748b; margin-top:0.3rem; text-transform:uppercase; letter-spacing:0.08em; }
.pred-banner { border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:0.6rem; background:linear-gradient(135deg,#0a0e1a,#0f172a); }
.food-item { background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:0.5rem 0.7rem; text-align:center; margin:0.15rem; display:inline-block; }
.food-item-name { font-size:0.62rem; color:#64748b; text-transform:uppercase; }
.food-item-qty { font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:700; color:#fbbf24; }
</style>
""", unsafe_allow_html=True)

CLEAN = "campus_data/cleaned"
MDIR  = "campus_data/models"

@st.cache_data
def load_all():
    att      = pd.read_csv(f"{CLEAN}/attendance_clean.csv",   parse_dates=["date"])
    wifi     = pd.read_csv(f"{CLEAN}/wifi_daily.csv",         parse_dates=["date"])
    elec     = pd.read_csv(f"{CLEAN}/electricity_with_anomalies.csv", parse_dates=["date"])
    mess     = pd.read_csv(f"{CLEAN}/mess_daily.csv",         parse_dates=["date"])
    food     = pd.read_csv(f"{MDIR}/food_quantity_estimates.csv",     parse_dates=["date"])
    master   = pd.read_csv(f"{CLEAN}/master_student_day.csv", parse_dates=["date"])
    students = pd.read_csv("campus_data/students_master.csv")
    return att, wifi, elec, mess, food, master, students

@st.cache_resource
def load_models():
    m = {}
    for f in os.listdir(MDIR):
        if f.endswith(".pkl"):
            with open(f"{MDIR}/{f}", "rb") as fh:
                m[f.replace(".pkl","")] = pickle.load(fh)
    return m

try:
    att, wifi, elec, mess, food, master, students = load_all()
    models = load_models()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Outfit", size=12),
    margin=dict(l=10, r=10, t=35, b=10),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    title_font=dict(color="#e2e8f0", size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)
COLORS = ["#6366f1","#22d3ee","#4ade80","#fbbf24","#f472b6","#f87171","#a78bfa"]
FOOD_MAP = {
    "Breakfast":{"Upma":200,"Tea":150,"Bread":100},
    "Lunch":    {"Rice":350,"Dal":200,"Sabzi":150,"Roti":120},
    "Snacks":   {"Tea":150,"Snack":100},
    "Dinner":   {"Rice":300,"Dal":200,"Sabzi":150,"Roti":120},
}
MEAL_COLORS = {"Breakfast":"#fbbf24","Lunch":"#4ade80","Snacks":"#22d3ee","Dinner":"#6366f1"}

# HERO
st.markdown('<div class="hero"><h1>🎓 Campus Analytics</h1><p>Attendance · WiFi · Electricity · Mess Predictions · ML Insights — all in one view</p></div>', unsafe_allow_html=True)

# FILTERS
fc1, fc2 = st.columns([2,2])
with fc1:
    dept_filter = st.multiselect("Department", options=students["department"].unique().tolist(),
        default=students["department"].unique().tolist(), placeholder="All Departments")
with fc2:
    date_range = st.date_input("Date Range",
        value=[att["date"].min().date(), att["date"].max().date()],
        min_value=att["date"].min().date(), max_value=att["date"].max().date())

if not dept_filter: dept_filter = students["department"].unique().tolist()
d1 = pd.Timestamp(date_range[0]) if len(date_range)>=1 else att["date"].min()
d2 = pd.Timestamp(date_range[1]) if len(date_range)==2 else att["date"].max()

att_f    = att[(att["date"].between(d1,d2)) & (att["department"].isin(dept_filter))]
wifi_f   = wifi[wifi["date"].between(d1,d2)]
elec_f   = elec[elec["date"].between(d1,d2)]
mess_f   = mess[mess["date"].between(d1,d2)]
master_f = master[master["date"].between(d1,d2)]
food_f   = food[food["date"].between(d1,d2)]

# KPIs
st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
avg_att   = att_f["status_bin"].mean()*100 if len(att_f)>0 else 0
avg_mb    = wifi_f["total_mb"].mean() if len(wifi_f)>0 else 0
total_kwh = elec_f["consumption_kwh"].sum() if len(elec_f)>0 else 0
avg_mess  = mess_f["footfall"].mean() if len(mess_f)>0 else 0
anomalies = int(elec_f["predicted_anomaly"].sum()) if "predicted_anomaly" in elec_f.columns else 0
at_risk   = int((master_f["daily_att"]<0.4).sum()) if "daily_att" in master_f.columns else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
for col,icon,val,label,color in [
    (k1,"📅",f"{avg_att:.1f}%","Avg Attendance","#6366f1"),
    (k2,"📶",f"{avg_mb:.0f} MB","WiFi/Student/Day","#22d3ee"),
    (k3,"⚡",f"{total_kwh:,.0f}","Total kWh","#fbbf24"),
    (k4,"🍽️",f"{avg_mess:.0f}","Avg Meal Footfall","#4ade80"),
    (k5,"🚨",f"{anomalies}","Power Anomalies","#f87171"),
    (k6,"⚠️",f"{at_risk:,}","At-Risk Records","#f472b6"),
]:
    col.markdown(f'<div class="kpi-card" style="--accent:{color}"><div class="kpi-icon">{icon}</div><div class="kpi-val">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

# MESS PREDICTIONS
st.markdown('<div class="section-title">🍽️ Today\'s Mess — Food Required (ML Predicted)</div>', unsafe_allow_html=True)
lf = food_f.sort_values("date", ascending=False)
if len(lf)>0:
    ld = lf["date"].iloc[0]
    ft = lf[lf["date"]==ld]
    m1,m2,m3,m4 = st.columns(4)
    for meal,col in zip(["Breakfast","Lunch","Snacks","Dinner"],[m1,m2,m3,m4]):
        sub = ft[ft["meal_type"]==meal]
        pf  = int(sub["pred_footfall"].iloc[0]) if len(sub)>0 else 0
        clr = MEAL_COLORS[meal]
        items = FOOD_MAP[meal]
        fhtml = "".join([f'<div class="food-item"><div class="food-item-name">{k}</div><div class="food-item-qty">{pf*v/1000:.1f}kg</div></div>' for k,v in items.items()])
        col.markdown(f'<div class="pred-banner" style="border:1px solid {clr}50"><div style="font-size:0.7rem;color:{clr};text-transform:uppercase;margin-bottom:0.4rem">● {meal}</div><div style="font-family:JetBrains Mono,monospace;font-size:2rem;font-weight:700;color:{clr}">{pf}</div><div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem">students expected</div>{fhtml}</div>', unsafe_allow_html=True)

# ATTENDANCE
st.markdown('<div class="section-title">📅 Attendance Analytics</div>', unsafe_allow_html=True)
a1,a2,a3 = st.columns([2,1,1])

daily = att_f.groupby("date")["status_bin"].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily["date"],y=daily["status_bin"]*100,fill="tozeroy",
    line=dict(color="#6366f1",width=2),fillcolor="rgba(99,102,241,0.12)",name="Attendance %"))
fig.add_hline(y=75,line_dash="dash",line_color="#f87171",
    annotation_text="75% min",annotation_font_color="#f87171")
fig.update_layout(**THEME,title="Daily Attendance Rate (%)",yaxis_range=[0,100])
a1.plotly_chart(fig,use_container_width=True)

sc = att_f["status"].value_counts().reset_index()
fig2 = go.Figure(go.Pie(labels=sc["status"],values=sc["count"],hole=0.6,
    marker_colors=["#4ade80","#f87171","#fbbf24"]))
fig2.update_layout(**THEME,title="Status Distribution",showlegend=True,
    annotations=[dict(text=f"{avg_att:.0f}%",x=0.5,y=0.5,font_size=18,font_color="#e2e8f0",showarrow=False)])
a2.plotly_chart(fig2,use_container_width=True)

da = att_f.groupby("department")["status_bin"].mean().sort_values().reset_index()
fig3 = go.Figure(go.Bar(x=da["status_bin"]*100,y=da["department"],orientation="h",
    marker_color=COLORS[:len(da)],text=[f"{v:.1f}%" for v in da["status_bin"]*100],textposition="outside"))
fig3.update_layout(**THEME,title="Attendance by Department",xaxis_range=[0,110])
a3.plotly_chart(fig3,use_container_width=True)

b1,b2 = st.columns([1,2])
subj = att_f.groupby("subject")["status_bin"].mean().sort_values().reset_index()
fig4 = go.Figure(go.Bar(x=subj["status_bin"]*100,y=subj["subject"],orientation="h",
    marker=dict(color=subj["status_bin"]*100,colorscale="Blues"),
    text=[f"{v:.1f}%" for v in subj["status_bin"]*100],textposition="outside"))
fig4.update_layout(**THEME,title="Attendance by Subject",xaxis_range=[0,110])
b1.plotly_chart(fig4,use_container_width=True)

hd = att_f.groupby(["department","day_of_week"])["status_bin"].mean().unstack().fillna(0)
do = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
hd = hd.reindex(columns=[d for d in do if d in hd.columns])
fig5 = px.imshow(hd*100,color_continuous_scale="Blues",labels=dict(color="Att %"),title="Heatmap: Dept × Day")
fig5.update_layout(**THEME)
b2.plotly_chart(fig5,use_container_width=True)

st.markdown('<div class="section-title">⚠️ At-Risk Students (Attendance &lt; 50%)</div>', unsafe_allow_html=True)
rdf = att_f.groupby("student_id").agg(att_rate=("status_bin","mean"),total=("status_bin","count"),absences=("status_bin",lambda x:(x==0).sum())).reset_index()
rdf = rdf[rdf["att_rate"]<0.5].sort_values("att_rate").head(15)
rdf["att_rate"] = (rdf["att_rate"]*100).round(1)
rdf["Risk"] = rdf["att_rate"].apply(lambda x:"🔴 Critical" if x<35 else "🟠 High")
rdf.columns=["Student ID","Att %","Classes","Absences","Risk"]
st.dataframe(rdf,use_container_width=True,hide_index=True,
    column_config={"Att %":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f%%")})

# WIFI
st.markdown('<div class="section-title">📶 WiFi Usage Analytics</div>', unsafe_allow_html=True)
w1,w2,w3 = st.columns(3)

wt = wifi_f.groupby("date")["total_mb"].mean().reset_index()
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=wt["date"],y=wt["total_mb"],fill="tozeroy",
    line=dict(color="#22d3ee",width=2),fillcolor="rgba(34,211,238,0.1)"))
fig6.update_layout(**THEME,title="Avg Daily WiFi Usage (MB)")
w1.plotly_chart(fig6,use_container_width=True)

st2 = wifi_f.groupby("date")["sessions"].mean().reset_index()
fig7 = go.Figure(go.Bar(x=st2["date"],y=st2["sessions"],marker_color="#6366f1",opacity=0.8))
fig7.update_layout(**THEME,title="Avg Sessions/Student/Day")
w2.plotly_chart(fig7,use_container_width=True)

if "total_mb" in master_f.columns and "daily_att" in master_f.columns:
    samp = master_f.dropna(subset=["total_mb","daily_att"]).sample(min(1500,len(master_f)))
    fig8 = px.scatter(samp,x="total_mb",y="daily_att",
        color="department" if "department" in samp.columns else None,
        opacity=0.5,color_discrete_sequence=COLORS,
        labels={"total_mb":"WiFi (MB)","daily_att":"Attendance Rate"})
    fig8.update_layout(**THEME,title="WiFi vs Attendance Correlation")
    w3.plotly_chart(fig8,use_container_width=True)

# ELECTRICITY
st.markdown('<div class="section-title">⚡ Electricity Consumption & Anomalies</div>', unsafe_allow_html=True)
e1,e2 = st.columns([2,1])
bld = elec_f.groupby(["date","building"])["consumption_kwh"].sum().reset_index()
fig9 = px.line(bld,x="date",y="consumption_kwh",color="building",color_discrete_sequence=COLORS,labels={"consumption_kwh":"kWh"})
fig9.update_layout(**THEME,title="Daily Consumption by Building")
e1.plotly_chart(fig9,use_container_width=True)
slot = elec_f.groupby("time_slot")["consumption_kwh"].mean().reset_index()
fig10 = go.Figure(go.Bar(x=slot["time_slot"],y=slot["consumption_kwh"],
    marker=dict(color=slot["consumption_kwh"],colorscale="YlOrRd"),
    text=[f"{v:.2f}" for v in slot["consumption_kwh"]],textposition="outside"))
fig10.update_layout(**THEME,title="Avg Consumption by Time Slot")
e2.plotly_chart(fig10,use_container_width=True)

e3,e4 = st.columns([2,1])
ne = elec_f[elec_f["predicted_anomaly"]==0]
ae = elec_f[elec_f["predicted_anomaly"]==1]
fig11 = go.Figure()
fig11.add_trace(go.Scatter(x=ne["date"],y=ne["consumption_kwh"],mode="markers",name="Normal",marker=dict(color="#22d3ee",size=3,opacity=0.3)))
fig11.add_trace(go.Scatter(x=ae["date"],y=ae["consumption_kwh"],mode="markers",name="⚠️ Anomaly",marker=dict(color="#f87171",size=9,symbol="x",line=dict(width=2))))
fig11.update_layout(**THEME,title="Anomaly Detection — All Buildings")
e3.plotly_chart(fig11,use_container_width=True)
at2 = elec_f[elec_f["predicted_anomaly"]==1][["date","building","time_slot","consumption_kwh","rolling_3d_avg"]].copy()
if len(at2)>0:
    at2["Excess %"]=((at2["consumption_kwh"]/at2["rolling_3d_avg"]-1)*100).round(1)
    at2=at2.sort_values("consumption_kwh",ascending=False).head(10)
    at2.columns=["Date","Building","Slot","kWh","3d Avg","Excess %"]
    e4.markdown("**⚠️ Top Anomalies**")
    e4.dataframe(at2.reset_index(drop=True),use_container_width=True,hide_index=True,height=280)

# MESS
st.markdown('<div class="section-title">🍽️ Mess Footfall Analytics</div>', unsafe_allow_html=True)
ms1,ms2,ms3 = st.columns(3)
fig12 = px.line(mess_f,x="date",y="footfall",color="meal_type",color_discrete_sequence=COLORS,labels={"footfall":"Students"})
fig12.update_layout(**THEME,title="Meal-wise Footfall Over Time")
ms1.plotly_chart(fig12,use_container_width=True)
dov=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dm=mess_f.groupby(["day_of_week","meal_type"])["footfall"].mean().reset_index()
dm["day_of_week"]=pd.Categorical(dm["day_of_week"],categories=dov,ordered=True)
dm=dm.sort_values("day_of_week")
fig13=px.bar(dm,x="day_of_week",y="footfall",color="meal_type",barmode="group",color_discrete_sequence=COLORS,labels={"footfall":"Students","day_of_week":"Day"})
fig13.update_layout(**THEME,title="Footfall by Day × Meal")
ms2.plotly_chart(fig13,use_container_width=True)
ec=mess_f.groupby(["meal_type","is_exam_week"])["footfall"].mean().reset_index()
ec["Period"]=ec["is_exam_week"].map({True:"Exam Week",False:"Regular"})
fig14=px.bar(ec,x="meal_type",y="footfall",color="Period",barmode="group",color_discrete_sequence=["#f87171","#4ade80"],labels={"footfall":"Avg Students"})
fig14.update_layout(**THEME,title="Exam vs Regular Week Footfall")
ms3.plotly_chart(fig14,use_container_width=True)

# ML PREDICTIONS
st.markdown('<div class="section-title">🤖 Live ML Predictions</div>', unsafe_allow_html=True)
p1,p2,p3 = st.columns(3)

with p1:
    st.markdown("**🎯 Attendance Risk Predictor**")
    ra=st.slider("7-day Attendance Rate",0.0,1.0,0.75,0.01,key="r1")
    mb=st.slider("Yesterday WiFi (MB)",0,2000,300,key="r2")
    ss=st.slider("WiFi Sessions",0,10,3,key="r3")
    dv=st.selectbox("Department",students["department"].unique(),key="r4")
    dy=st.selectbox("Day",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],key="r5")
    if st.button("🔮 Predict Risk",use_container_width=True):
        try:
            ld=models["label_enc_dept"]; lw=models["label_enc_day"]
            de=ld.transform([dv])[0] if dv in ld.classes_ else 0
            we=lw.transform([dy])[0] if dy in lw.classes_ else 0
            X=np.array([[ra,mb,ss,de,we,1 if dy=="Saturday" else 0,10,2]])
            pred=models["attendance_risk_model"].predict(X)[0]
            prob=models["attendance_risk_model"].predict_proba(X)[0]
            clr={"High Risk":"#f87171","Medium Risk":"#fbbf24","Low Risk":"#4ade80"}.get(pred,"#4ade80")
            st.markdown(f'<div style="background:#0f172a;border:1px solid {clr}50;border-radius:10px;padding:1rem;text-align:center;margin-top:0.5rem"><div style="font-size:0.7rem;color:#64748b">PREDICTION</div><div style="font-size:1.8rem;font-weight:800;color:{clr}">{pred}</div><div style="font-size:0.75rem;color:#64748b">Confidence: {max(prob)*100:.1f}%</div></div>',unsafe_allow_html=True)
        except Exception as ex: st.error(str(ex))

with p2:
    st.markdown("**🍽️ Mess Footfall + Food Needed**")
    ml=st.selectbox("Meal",["Breakfast","Lunch","Snacks","Dinner"],key="ml1")
    md=st.selectbox("Day",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],key="ml2")
    ex=st.checkbox("Exam Week?",key="ml3")
    pv=st.number_input("Prev Day Footfall",50,500,350,key="ml4")
    rv=st.number_input("7-day Avg Footfall",50,500,340,key="ml5")
    if st.button("🍽️ Predict Footfall",use_container_width=True):
        try:
            lm=models["label_enc_meal"]; lmd=models["label_enc_mess_day"]
            me=lm.transform([ml])[0] if ml in lm.classes_ else 0
            mde=lmd.transform([md])[0] if md in lmd.classes_ else 0
            Xm=np.array([[me,mde,int(ex),pv,rv,2,10]])
            pf=int(models["mess_footfall_model"].predict(Xm)[0])
            items=FOOD_MAP.get(ml,{})
            fhtml="".join([f'<div class="food-item"><div class="food-item-name">{k}</div><div class="food-item-qty">{pf*v/1000:.1f}kg</div></div>' for k,v in items.items()])
            st.markdown(f'<div style="background:#0f172a;border:1px solid #4ade8050;border-radius:10px;padding:1rem;margin-top:0.5rem"><div style="font-size:0.7rem;color:#64748b">EXPECTED STUDENTS</div><div style="font-size:2rem;font-weight:800;color:#4ade80">{pf}</div><div style="font-size:0.75rem;color:#64748b;margin:0.4rem 0">Food Required:</div><div>{fhtml}</div></div>',unsafe_allow_html=True)
        except Exception as ex: st.error(str(ex))

with p3:
    st.markdown("**📶 WiFi Usage Predictor**")
    l1=st.number_input("Yesterday (MB)",0,3000,300,key="w1")
    l7=st.number_input("7-days-ago (MB)",0,3000,280,key="w2")
    sw=st.number_input("Sessions",0,20,3,key="w3")
    dw=st.selectbox("Department",students["department"].unique(),key="w4")
    dow=st.selectbox("Tomorrow",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],key="w5")
    if st.button("📶 Predict Usage",use_container_width=True):
        try:
            ld2=models["label_enc_dept"]
            de2=ld2.transform([dw])[0] if dw in ld2.classes_ else 0
            dm2={"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            Xw=np.array([[l1,l7,sw,de2,dm2.get(dow,0),10,2]])
            pw=models["wifi_usage_model"].predict(Xw)[0]
            st.markdown(f'<div style="background:#0f172a;border:1px solid #22d3ee50;border-radius:10px;padding:1rem;text-align:center;margin-top:0.5rem"><div style="font-size:0.7rem;color:#64748b">PREDICTED USAGE</div><div style="font-size:2rem;font-weight:800;color:#22d3ee">{pw:.0f} MB</div><div style="font-size:0.85rem;color:#64748b">= {pw/1024:.2f} GB</div></div>',unsafe_allow_html=True)
        except Exception as ex: st.error(str(ex))

# FEATURE IMPORTANCE
st.markdown('<div class="section-title">📈 Model Feature Importance</div>', unsafe_allow_html=True)
fi1,fi2=st.columns(2)
try:
    af=pd.read_csv(f"{MDIR}/attendance_feature_importance.csv",names=["Feature","Importance"],header=0).sort_values("Importance")
    fig15=go.Figure(go.Bar(x=af["Importance"],y=af["Feature"],orientation="h",
        marker=dict(color=af["Importance"],colorscale="Blues"),
        text=[f"{v:.3f}" for v in af["Importance"]],textposition="outside"))
    fig15.update_layout(**THEME,title="Attendance Model Features")
    fi1.plotly_chart(fig15,use_container_width=True)
except: pass
try:
    mf=pd.read_csv(f"{MDIR}/mess_feature_importance.csv",names=["Feature","Importance"],header=0).sort_values("Importance")
    fig16=go.Figure(go.Bar(x=mf["Importance"],y=mf["Feature"],orientation="h",
        marker=dict(color=mf["Importance"],colorscale="Greens"),
        text=[f"{v:.3f}" for v in mf["Importance"]],textposition="outside"))
    fig16.update_layout(**THEME,title="Mess Footfall Model Features")
    fi2.plotly_chart(fig16,use_container_width=True)
except: pass

st.markdown('<div style="text-align:center;padding:2rem 0 1rem;color:#334155;font-size:0.75rem">Campus Analytics · Streamlit + Plotly · Simulated Data</div>', unsafe_allow_html=True)
