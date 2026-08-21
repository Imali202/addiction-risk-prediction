import streamlit as st
import os
from config import (
    ASSETS_DIR, RISK_TIER_LABELS,
    GENDER_OPTIONS, SCHOOL_GRADE_OPTIONS, PURPOSE_OPTIONS
)
from utils import load_artifacts, predict_risk, load_css

st.set_page_config(page_title="Digital Addiction Risk Screener", page_icon="📱", layout="wide")
load_css(os.path.join(ASSETS_DIR, "style.css"))

st.title("📱 Digital Addiction Risk Screener")
st.caption(
    "A public-health screening aid based on self-reported behavior. "
    "This is **not a diagnostic tool** — high-risk results should be followed up with a professional."
)

classifier, regressor, pipeline, schema = load_artifacts()

with st.form("risk_form"):
    st.subheader("Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 13, 19, 15)
    with c2:
        gender = st.selectbox("Gender", GENDER_OPTIONS)
    with c3:
        school_grade = st.selectbox("School Grade", SCHOOL_GRADE_OPTIONS)

    st.subheader("Usage Patterns")
    c1, c2, c3 = st.columns(3)
    with c1:
        daily_usage = st.slider("Daily Usage (hrs)", 0.0, 16.0, 4.0, 0.1)
        weekend_usage = st.slider("Weekend Usage (hrs)", 0.0, 16.0, 5.0, 0.1)
    with c2:
        phone_checks = st.number_input("Phone Checks Per Day", 0, 300, 50)
        apps_used = st.number_input("Apps Used Daily", 0, 30, 5)
    with c3:
        screen_before_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0, 0.1)
        purpose = st.selectbox("Primary Usage Purpose", PURPOSE_OPTIONS)

    st.subheader("Time Breakdown")
    c1, c2, c3 = st.columns(3)
    with c1:
        time_social = st.slider("Time on Social Media (hrs)", 0.0, 10.0, 1.5, 0.1)
    with c2:
        time_gaming = st.slider("Time on Gaming (hrs)", 0.0, 10.0, 1.0, 0.1)
    with c3:
        time_education = st.slider("Time on Education (hrs)", 0.0, 10.0, 1.0, 0.1)

    st.subheader("Wellbeing & Lifestyle")
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
        exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 1.0, 0.1)
    with c2:
        anxiety = st.slider("Anxiety Level (0-10)", 0, 10, 3)
        depression = st.slider("Depression Level (0-10)", 0, 10, 3)
    with c3:
        self_esteem = st.slider("Self Esteem (0-10)", 0, 10, 5)
        social_interactions = st.slider("Social Interactions (0-10)", 0, 10, 5)

    st.subheader("Family & Academics")
    c1, c2, c3 = st.columns(3)
    with c1:
        family_comm = st.slider("Family Communication (0-10)", 0, 10, 5)
    with c2:
        parental_control = st.selectbox("Parental Control", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c3:
        academic_perf = st.slider("Academic Performance (0-100)", 0, 100, 75)

    submitted = st.form_submit_button("Get Risk Assessment", use_container_width=True)

if submitted:
    raw_form_values = {
        "Age": age,
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep_hours,
        "Exercise_Hours": exercise_hours,
        "Screen_Time_Before_Bed": screen_before_bed,
        "Phone_Checks_Per_Day": phone_checks,
        "Apps_Used_Daily": apps_used,
        "Time_on_Social_Media": time_social,
        "Time_on_Gaming": time_gaming,
        "Time_on_Education": time_education,
        "Weekend_Usage_Hours": weekend_usage,
        "Anxiety_Level": anxiety,
        "Depression_Level": depression,
        "Self_Esteem": self_esteem,
        "Social_Interactions": social_interactions,
        "Family_Communication": family_comm,
        "Academic_Performance": academic_perf,
        "Gender": gender,
        "School_Grade": school_grade,
        "Phone_Usage_Purpose": purpose,
        "Parental_Control": parental_control,
    }

    try:
        risk_tier_pred, score, X_debug = predict_risk(raw_form_values)
        label = RISK_TIER_LABELS.get(risk_tier_pred, str(risk_tier_pred))
        css_class = f"risk-{str(label).lower()}"

        st.markdown(
            f'<div class="card">'
            f'<span class="risk-badge {css_class}">{label} Risk</span>'
            f'&nbsp;&nbsp; Predicted Addiction Score: <strong>{score:.1f}/10</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        if label == "High":
            st.warning("This result suggests elevated risk. Consider following up with a school counselor or clinician.")

        with st.expander("See input summary"):
            st.dataframe(X_debug.T.rename(columns={0: "Value"}))

    except ValueError as e:
        st.error(f"Prediction error: {e}")