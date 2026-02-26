import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import random, os

st.set_page_config(page_title="Snowflake Healthcare Analytics", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    .kpi-card { background: linear-gradient(135deg, #FFF 0%, #F0F4F8 100%); border-left: 4px solid #0077B6;
        border-radius: 8px; padding: 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .kpi-card h4 { margin:0; font-size:0.8rem; color:#5A6C7D; text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-card .val { font-size:1.7rem; font-weight:700; color:#1B2A4A; margin:0.1rem 0 0 0; }
    .kpi-green { border-left-color: #06D6A0; } .kpi-red { border-left-color: #EF476F; }
    .kpi-orange { border-left-color: #FFD166; } .kpi-teal { border-left-color: #48CAE4; }
    .arch-box { background: linear-gradient(135deg, #0077B6, #1B2A4A); color: white;
        border-radius: 12px; padding: 1.5rem; text-align: center; }
    .pass-badge { background:#E8F5E9; color:#2E7D32; padding:2px 10px; border-radius:12px; font-weight:600; }
    .fail-badge { background:#FFEBEE; color:#C62828; padding:2px 10px; border-radius:12px; font-weight:600; }
    div[data-testid="stMetric"] { background-color: #F0F4F8; border: 1px solid #D2D2D7; padding: 12px; border-radius: 10px; }
</style>""", unsafe_allow_html=True)

def kpi(title, value, css=""):
    st.markdown(f'<div class="kpi-card {css}"><h4>{title}</h4><p class="val">{value}</p></div>', unsafe_allow_html=True)

# ─── Data Generation ───
ICD10 = [("I21.0","Acute MI anterior","Cardiology"),("I50.9","Heart failure","Cardiology"),
    ("I10","Hypertension","Cardiology"),("J18.9","Pneumonia","Pulmonology"),
    ("J44.1","COPD exacerbation","Pulmonology"),("E11.9","Type 2 diabetes","Internal Medicine"),
    ("N17.9","Acute kidney failure","Internal Medicine"),("S72.001A","Hip fracture","Orthopedics"),
    ("G43.909","Migraine","Neurology"),("I63.9","Cerebral infarction","Neurology"),
    ("C34.90","Lung cancer","Oncology"),("C50.919","Breast cancer","Oncology"),
    ("A41.9","Sepsis","Emergency"),("K35.80","Appendicitis","General Surgery"),
    ("K80.0","Cholecystitis","Gastroenterology")]

DEPTS = ["Emergency","Cardiology","Orthopedics","Neurology","Oncology","Pediatrics","General Surgery","Internal Medicine","Pulmonology","Gastroenterology"]
ADM_TYPES = ["Emergency","Urgent","Elective","Newborn","Trauma"]
DISPOSITIONS = ["Home","Home Health","SNF","Transferred","AMA","Expired"]
INSURERS = ["BCBS","UnitedHealthcare","Aetna","Cigna","Humana","Kaiser","Medicare","Medicaid"]

@st.cache_data
def generate_healthcare_data(n_patients=2000):
    np.random.seed(42); random.seed(42)
    # Patients
    genders = np.random.choice(["M","F"], n_patients)
    ages = np.random.randint(1, 95, n_patients)
    patients = pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(1, n_patients+1)],
        "age": ages, "gender": genders,
        "race": np.random.choice(["White","Black","Hispanic","Asian","Other"], n_patients),
        "insurance": np.random.choice(INSURERS, n_patients),
        "city": np.random.choice(["New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","Dallas","Denver","Seattle"], n_patients),
    })
    # Encounters
    enc_rows = []
    enc_id = 1
    for _, pat in patients.iterrows():
        n_enc = max(1, int(np.random.poisson(3)))
        for _ in range(n_enc):
            admit = datetime(2023,1,1) + timedelta(days=random.randint(0,730))
            los = max(1, int(np.random.lognormal(1.2, 0.8)))
            diag = random.choice(ICD10)
            adm_type = random.choices(ADM_TYPES, weights=[30,15,40,5,10])[0]
            readmit_prob = 0.12 + (0.08 if los > 7 else 0) + (0.06 if adm_type == "Emergency" else 0)
            charges = round(random.uniform(2000, 120000), 2)
            enc_rows.append({
                "encounter_id": f"E{enc_id:07d}", "patient_id": pat["patient_id"],
                "admit_date": admit.strftime("%Y-%m-%d"),
                "discharge_date": (admit + timedelta(days=los)).strftime("%Y-%m-%d"),
                "length_of_stay": los, "department": diag[2],
                "admission_type": adm_type,
                "discharge_disposition": random.choices(DISPOSITIONS, weights=[50,15,15,10,5,5])[0],
                "icd10_code": diag[0], "diagnosis_desc": diag[1], "diagnosis_category": diag[2],
                "is_readmission": random.random() < readmit_prob,
                "total_charges": charges,
                "insurance_paid": round(charges * random.uniform(0.6, 0.9), 2),
            })
            enc_id += 1
    encounters = pd.DataFrame(enc_rows)
    encounters["patient_paid"] = (encounters["total_charges"] - encounters["insurance_paid"]).round(2)
    return patients, encounters

patients_df, encounters_df = generate_healthcare_data()

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### 🏥 Healthcare Analytics")
    st.markdown("*Snowflake + AWS*")
    st.markdown("---")
    page = st.radio("", [
        "🏗️ Architecture Overview",
        "👤 Patient Data Explorer",
        "📊 Clinical Analytics",
        "✅ Data Quality & Compliance",
        "⚙️ ETL Pipeline Monitor",
        "🤖 Predictive Analytics",
    ], label_visibility="collapsed")
    st.markdown("---")
    enc_dates = pd.to_datetime(encounters_df["admit_date"])
    date_range = st.date_input("Date Range", value=(enc_dates.min().date(), enc_dates.max().date()))
    selected_depts = st.multiselect("Department", sorted(encounters_df["department"].unique()), default=sorted(encounters_df["department"].unique()))

filtered_enc = encounters_df.copy()
filtered_enc = filtered_enc[filtered_enc["department"].isin(selected_depts)]
if len(date_range) == 2:
    dates = pd.to_datetime(filtered_enc["admit_date"])
    filtered_enc = filtered_enc[(dates >= pd.Timestamp(date_range[0])) & (dates <= pd.Timestamp(date_range[1]))]

# ════════════════════════════════
if page == "🏗️ Architecture Overview":
    st.markdown("## End-to-End Snowflake Healthcare Analytics Architecture")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Patients", f"{len(patients_df):,}")
    c2.metric("Encounters", f"{len(encounters_df):,}")
    c3.metric("Data Layers", "3 (Bronze/Silver/Gold)")
    c4.metric("ML Models", "1 (Readmission)")

    # Architecture diagram
    fig = go.Figure()
    nodes = [
        (0,3,"EHR System","#EF476F"), (0,2,"Claims Data","#EF476F"), (0,1,"Lab Results","#EF476F"),
        (1.5,2,"Amazon S3","#FF9F1C"),
        (3,2,"BRONZE\n(Raw)","#48CAE4"), (4.5,2,"SILVER\n(Curated)","#00B4D8"), (6,2,"GOLD\n(Analytics)","#0077B6"),
        (7.5,3,"Dashboards","#06D6A0"), (7.5,2,"ML Models","#06D6A0"), (7.5,1,"Reporting","#06D6A0"),
    ]
    for x,y,label,color in nodes:
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
            marker=dict(size=50,color=color,opacity=0.9,line=dict(width=2,color="white")),
            text=[label],textposition="middle center",textfont=dict(size=9,color="white"),showlegend=False))
    for x0,y0,x1,y1 in [(0,3,1.5,2),(0,2,1.5,2),(0,1,1.5,2),(1.5,2,3,2),(3,2,4.5,2),(4.5,2,6,2),(6,2,7.5,3),(6,2,7.5,2),(6,2,7.5,1)]:
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowsize=1.5,arrowwidth=2,arrowcolor="#8899AA")
    fig.update_layout(xaxis=dict(range=[-0.8,8.3],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(range=[0,4],showgrid=False,zeroline=False,showticklabels=False),
        height=380,margin=dict(l=10,r=10,t=10,b=10),plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown('<div class="arch-box" style="background:linear-gradient(135deg,#48CAE4,#0096C7);"><h3 style="color:white;margin:0;">BRONZE</h3><p style="color:#E0F7FA;">Raw / Staging</p><p style="color:white;font-size:0.85rem;text-align:left;">- Snowpipe auto-ingest from S3<br>- Schema-on-read with VARIANT<br>- Full CDC tracking</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="arch-box" style="background:linear-gradient(135deg,#00B4D8,#0077B6);"><h3 style="color:white;margin:0;">SILVER</h3><p style="color:#E0F7FA;">Curated / Clean</p><p style="color:white;font-size:0.85rem;text-align:left;">- dbt transformations<br>- PII masking (SHA-256)<br>- ICD-10 code validation</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="arch-box"><h3 style="color:white;margin:0;">GOLD</h3><p style="color:#E0F7FA;">Analytics</p><p style="color:white;font-size:0.85rem;text-align:left;">- Star schema fact/dim tables<br>- Readmission risk scores<br>- Pre-aggregated KPI views</p></div>', unsafe_allow_html=True)

# ════════════════════════════════
elif page == "👤 Patient Data Explorer":
    st.markdown("## Patient Data Explorer")
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Total Patients", f"{len(patients_df):,}", "kpi-teal")
    with k2: kpi("Total Encounters", f"{len(filtered_enc):,}")
    with k3: kpi("Avg LOS", f"{filtered_enc['length_of_stay'].mean():.1f} days", "kpi-green")
    with k4: kpi("Readmission Rate", f"{filtered_enc['is_readmission'].mean()*100:.1f}%", "kpi-orange")
    with k5: kpi("Avg Charges", f"${filtered_enc['total_charges'].mean():,.0f}", "kpi-red")

    # Volume trends
    vol = filtered_enc.copy()
    vol["month"] = pd.to_datetime(vol["admit_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = vol.groupby("month")["patient_id"].nunique().reset_index()
    monthly.columns = ["month","patients"]
    fig = px.area(monthly, x="month", y="patients", title="Monthly Unique Patients", color_discrete_sequence=["#0077B6"])
    fig.update_traces(line=dict(width=2.5), fillcolor="rgba(0,119,182,0.1)")
    fig.update_layout(height=350, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Demographics
    c1,c2,c3 = st.columns(3)
    with c1:
        fig = px.pie(patients_df["gender"].value_counts().reset_index(), values="count", names="gender",
                     title="Gender", color_discrete_sequence=["#0077B6","#EF476F"])
        fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(patients_df["race"].value_counts().reset_index(), x="count", y="race", orientation="h",
                     title="Race", color_discrete_sequence=["#48CAE4"])
        fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.bar(patients_df["insurance"].value_counts().head(8).reset_index(), x="count", y="insurance",
                     orientation="h", title="Top Insurers", color_discrete_sequence=["#00B4D8"])
        fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)

    st.dataframe(patients_df.head(100), use_container_width=True, height=300)

# ════════════════════════════════
elif page == "📊 Clinical Analytics":
    st.markdown("## Clinical Analytics Dashboard")
    readmit_pct = filtered_enc["is_readmission"].mean()*100
    avg_los = filtered_enc["length_of_stay"].mean()
    mort_rate = (filtered_enc["discharge_disposition"]=="Expired").mean()*100
    total_rev = filtered_enc["total_charges"].sum()

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Readmission Rate", f"{readmit_pct:.1f}%", "kpi-green" if readmit_pct < 12 else "kpi-red")
    with k2: kpi("Avg LOS", f"{avg_los:.1f} days")
    with k3: kpi("Mortality Rate", f"{mort_rate:.2f}%", "kpi-red")
    with k4: kpi("Total Revenue", f"${total_rev/1e6:.1f}M", "kpi-teal")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        readmit = filtered_enc.groupby("department")["is_readmission"].mean().reset_index()
        readmit["rate"] = (readmit["is_readmission"]*100).round(2)
        fig = px.bar(readmit.sort_values("rate"), x="rate", y="department", orientation="h",
                     color="rate", color_continuous_scale=["#06D6A0","#FFD166","#EF476F"],
                     title="Readmission Rate by Department")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        top_diag = filtered_enc["diagnosis_desc"].value_counts().head(10).reset_index()
        top_diag.columns = ["diagnosis","count"]
        fig = px.bar(top_diag, x="count", y="diagnosis", orientation="h", title="Top 10 Diagnoses",
                     color_discrete_sequence=["#0077B6"])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        los_diag = filtered_enc.groupby("diagnosis_desc")["length_of_stay"].mean().reset_index()
        los_diag.columns = ["diagnosis","avg_los"]
        los_diag = los_diag.sort_values("avg_los", ascending=False).head(12)
        fig = px.bar(los_diag.sort_values("avg_los"), x="avg_los", y="diagnosis", orientation="h",
                     color="avg_los", color_continuous_scale=["#48CAE4","#0077B6","#1B2A4A"],
                     title="Avg LOS by Diagnosis")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        cost = filtered_enc.groupby("department")["total_charges"].mean().reset_index()
        cost.columns = ["department","avg_cost"]
        fig = px.bar(cost.sort_values("avg_cost"), x="avg_cost", y="department", orientation="h",
                     color="avg_cost", color_continuous_scale=["#06D6A0","#FFD166","#EF476F"],
                     title="Cost per Patient by Department",
                     text=cost.sort_values("avg_cost")["avg_cost"].apply(lambda x: f"${x:,.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Admission trends
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        enc_t = filtered_enc.copy()
        enc_t["month"] = pd.to_datetime(enc_t["admit_date"]).dt.to_period("M").dt.to_timestamp()
        enc_t["adm_group"] = enc_t["admission_type"].apply(lambda x: "Emergency/Urgent" if x in ("Emergency","Urgent","Trauma") else "Scheduled")
        adm = enc_t.groupby(["month","adm_group"]).size().reset_index(name="count")
        fig = px.line(adm, x="month", y="count", color="adm_group",
                      color_discrete_map={"Emergency/Urgent":"#EF476F","Scheduled":"#0077B6"},
                      title="Admission Trends")
        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with r3c2:
        enc_t["is_expired"] = enc_t["discharge_disposition"]=="Expired"
        mort = enc_t.groupby("month").agg(total=("encounter_id","count"),expired=("is_expired","sum")).reset_index()
        mort["rate"] = (mort["expired"]/mort["total"]*100).round(2)
        fig = px.line(mort, x="month", y="rate", title="Mortality Rate Trend", color_discrete_sequence=["#EF476F"])
        fig.update_traces(line=dict(width=2.5), fill="tonexty", fillcolor="rgba(239,71,111,0.08)")
        fig.add_hline(y=mort["rate"].mean(), line_dash="dash", line_color="#8899AA",
                      annotation_text=f"Avg: {mort['rate'].mean():.2f}%")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════
elif page == "✅ Data Quality & Compliance":
    st.markdown("## Data Quality & HIPAA Compliance")

    st.markdown("### PII Masking Demonstration")
    c1,c2 = st.columns(2)
    sample = patients_df.head(5)
    with c1:
        st.markdown("**Admin View (Full Access)**")
        st.dataframe(sample, use_container_width=True)
    with c2:
        st.markdown("**Analyst View (Masked)**")
        masked = sample.copy()
        masked["patient_id"] = masked["patient_id"].str[:2] + "****"
        masked["city"] = "***"
        st.dataframe(masked, use_container_width=True)

    st.markdown("### Data Completeness")
    tables = {"patients": patients_df, "encounters": encounters_df}
    comp_rows = []
    for name, df in tables.items():
        completeness = (1 - df.isnull().mean().mean()) * 100
        comp_rows.append({"table": name, "completeness": round(completeness, 2)})
    comp_df = pd.DataFrame(comp_rows)
    fig = px.bar(comp_df, x="table", y="completeness", color="completeness",
                 color_continuous_scale=["#EF476F","#FFD166","#06D6A0"], range_y=[90,100],
                 text="completeness", title="Table Completeness")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Referential Integrity")
    enc_patients = set(encounters_df["patient_id"].unique())
    all_patients = set(patients_df["patient_id"].unique())
    orphans = len(enc_patients - all_patients)
    st.markdown(f'<span class="pass-badge">PASS</span> encounters.patient_id → patients.patient_id — {orphans} orphan records', unsafe_allow_html=True)

    st.markdown("### Audit Trail")
    audit = pd.DataFrame({
        "timestamp": pd.date_range("2025-12-01", periods=20, freq="4h"),
        "user": [f"user_{np.random.randint(1,10):03d}@healthsys.com" for _ in range(20)],
        "action": np.random.choice(["Record viewed","PII accessed","Bulk query","Report generated","Data exported"], 20),
        "severity": np.random.choice(["LOW","LOW","HIGH","LOW"], 20),
    })
    st.dataframe(audit.sort_values("timestamp", ascending=False), use_container_width=True, height=300)

# ════════════════════════════════
elif page == "⚙️ ETL Pipeline Monitor":
    st.markdown("## ETL Pipeline Monitor")

    pipelines = ["ehr_patient_ingestion","claims_daily_load","diagnosis_transform","medication_reconciliation",
                 "analytics_cube_refresh","data_quality_scan","dbt_staging_run","dbt_curated_run"]
    np.random.seed(42)
    pipe_hist = []
    for day in range(60):
        dt = datetime(2025,10,1) + timedelta(days=day)
        for p in pipelines:
            success = np.random.random() > 0.05
            pipe_hist.append({"pipeline":p, "date":dt.strftime("%Y-%m-%d"),
                              "status":"SUCCESS" if success else "FAILED",
                              "duration": round(np.random.lognormal(3,0.5) if success else np.random.uniform(1,30),1),
                              "rows": int(np.random.lognormal(8,1.5)) if success else 0})
    pipe_df = pd.DataFrame(pipe_hist)

    total = len(pipe_df); success = (pipe_df["status"]=="SUCCESS").sum()
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Runs (60d)", f"{total:,}")
    with c2: kpi("Success Rate", f"{success/total*100:.1f}%", "kpi-green")
    with c3: kpi("Failed Runs", f"{total-success}", "kpi-red")
    with c4: kpi("Avg Duration", f"{pipe_df[pipe_df['status']=='SUCCESS']['duration'].mean():.0f}s", "kpi-teal")

    # Heatmap
    pivot = pipe_df.pivot_table(index="pipeline", columns="date", values="status",
                                 aggfunc=lambda x: 1 if all(v=="SUCCESS" for v in x) else 0)
    last30 = pivot[pivot.columns[-30:]]
    fig = px.imshow(last30, color_continuous_scale=[[0,"#EF476F"],[1,"#06D6A0"]], aspect="auto",
                    labels={"color":"Status"})
    fig.update_layout(height=350, coloraxis_showscale=False, xaxis_title="Date", yaxis_title="",
                      margin=dict(l=10,r=10,t=10,b=40))
    fig.update_xaxes(tickangle=45, dtick=3)
    st.plotly_chart(fig, use_container_width=True)

    # dbt DAG
    st.markdown("### dbt Transformation DAG")
    fig = go.Figure()
    dag = [(0,3,"stg_patients","#48CAE4"),(0,2,"stg_encounters","#48CAE4"),(0,1,"stg_claims","#48CAE4"),
           (1.5,3,"int_demographics","#00B4D8"),(1.5,2,"int_encounters","#00B4D8"),(1.5,1,"int_claims","#00B4D8"),
           (3,3,"dim_patient","#0077B6"),(3,2,"fct_encounter","#0077B6"),(3,1,"fct_claims","#0077B6"),
           (4.5,2.5,"mart_readmission","#1B2A4A"),(4.5,1.5,"mart_cost","#1B2A4A")]
    for x,y,label,color in dag:
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
            marker=dict(size=35,color=color,symbol="square",line=dict(width=1,color="white")),
            text=[label],textposition="middle center",textfont=dict(size=8,color="white"),showlegend=False))
    for x0,y0,x1,y1 in [(0,3,1.5,3),(0,2,1.5,2),(0,1,1.5,1),(1.5,3,3,3),(1.5,2,3,2),(1.5,1,3,1),
                          (3,3,4.5,2.5),(3,2,4.5,2.5),(3,2,4.5,1.5),(3,1,4.5,1.5)]:
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",
                           showarrow=True,arrowhead=2,arrowsize=1,arrowwidth=1.5,arrowcolor="#B0BEC5")
    fig.update_layout(xaxis=dict(range=[-0.6,5.3],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(range=[0,4],showgrid=False,zeroline=False,showticklabels=False),
        height=320,margin=dict(l=10,r=10,t=10,b=10),plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════
elif page == "🤖 Predictive Analytics":
    st.markdown("## Predictive Analytics — Readmission Risk Model")

    @st.cache_data
    def train_model():
        df = encounters_df.copy()
        le_dept = LabelEncoder(); le_adm = LabelEncoder(); le_disp = LabelEncoder(); le_diag = LabelEncoder()
        df["dept_enc"] = le_dept.fit_transform(df["department"])
        df["adm_enc"] = le_adm.fit_transform(df["admission_type"])
        df["disp_enc"] = le_disp.fit_transform(df["discharge_disposition"])
        df["diag_enc"] = le_diag.fit_transform(df["diagnosis_category"])
        X = df[["length_of_stay","total_charges","dept_enc","adm_enc","disp_enc","diag_enc"]].values
        y = df["is_readmission"].astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        features = ["Length of Stay","Total Charges","Department","Admission Type","Discharge Disposition","Diagnosis Category"]
        importance = pd.DataFrame({"feature": features, "importance": np.abs(model.coef_[0])}).sort_values("importance")
        metrics = {"auc": round(roc_auc_score(y_test, y_prob),4),
                   "precision": round(precision_score(y_test, y_pred, zero_division=0),4),
                   "recall": round(recall_score(y_test, y_pred, zero_division=0),4),
                   "f1": round(f1_score(y_test, y_pred, zero_division=0),4),
                   "cm": confusion_matrix(y_test, y_pred)}
        results = pd.DataFrame({"actual": y_test, "prob": y_prob})
        results["risk"] = pd.cut(results["prob"], bins=[0,0.15,0.35,1], labels=["Low","Medium","High"])
        return importance, metrics, results

    importance, metrics, results = train_model()

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("AUC-ROC", f"{metrics['auc']:.3f}", "kpi-green" if metrics["auc"] > 0.7 else "kpi-orange")
    with k2: kpi("Precision", f"{metrics['precision']:.3f}")
    with k3: kpi("Recall", f"{metrics['recall']:.3f}", "kpi-teal")
    with k4: kpi("F1 Score", f"{metrics['f1']:.3f}", "kpi-orange")

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(importance, x="importance", y="feature", orientation="h", color="importance",
                     color_continuous_scale=["#48CAE4","#0077B6","#1B2A4A"], title="Feature Importance")
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(results, x="prob", color="risk", nbins=50, barmode="overlay",
                           color_discrete_map={"Low":"#06D6A0","Medium":"#FFD166","High":"#EF476F"},
                           title="Risk Score Distribution")
        fig.update_traces(opacity=0.75)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.imshow(metrics["cm"], text_auto=True, color_continuous_scale=["#FFF","#0077B6"],
                        x=["No Readmit","Readmit"], y=["No Readmit","Readmit"], title="Confusion Matrix")
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tier_sum = results.groupby("risk").agg(count=("actual","count"), rate=("actual","mean")).reset_index()
        tier_sum["rate"] = (tier_sum["rate"]*100).round(1)
        st.markdown("### Risk Stratification")
        st.dataframe(tier_sum.rename(columns={"risk":"Risk Tier","count":"Patients","rate":"Actual Readmit %"}), use_container_width=True)
        fig = make_subplots(rows=1,cols=3,specs=[[{"type":"indicator"}]*3], subplot_titles=["Low","Medium","High"])
        colors = {"Low":"#06D6A0","Medium":"#FFD166","High":"#EF476F"}
        for i,tier in enumerate(["Low","Medium","High"]):
            val = tier_sum[tier_sum["risk"]==tier]["rate"].values[0] if tier in tier_sum["risk"].values else 0
            fig.add_trace(go.Indicator(mode="gauge+number", value=val, number={"suffix":"%","font":{"size":18}},
                gauge={"axis":{"range":[0,50]},"bar":{"color":colors[tier]},"bgcolor":"#F0F4F8"}), row=1, col=i+1)
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
