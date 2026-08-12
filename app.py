import pickle
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model/random_forest.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.caption("Random Forest Regression implemented completely from scratch.")

st.write(
    "Enter student information below to estimate the final exam score."
)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours / Day", 0.0, 12.0, 5.0, 0.5)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
    previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 70.0, 1.0)

with col2:
    assignments = st.slider("Assignments Completed", 0, 10, 7)
    sleep_hours = st.slider("Sleep Hours / Day", 0.0, 12.0, 7.0, 0.5)
    extracurricular = st.selectbox(
        "Extracurricular Activities", ["No", "Yes"]
    )

if st.button("Predict Performance", type="primary"):
    extra = 1 if extracurricular == "Yes" else 0

    input_data = np.array([[
        study_hours,
        attendance,
        previous_score,
        assignments,
        sleep_hours,
        extra
    ]], dtype=float)

    prediction = float(np.clip(model.predict(input_data)[0], 0, 100))

    st.metric("Predicted Final Score", f"{prediction:.1f} / 100")

    if prediction >= 80:
        st.success("Performance: Excellent 🌟")
    elif prediction >= 60:
        st.info("Performance: Good 👍")
    elif prediction >= 40:
        st.warning("Performance: Average 📚")
    else:
        st.error("Performance: Needs Improvement 💪")

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance_df.set_index("Feature")["Importance"]
    )

st.divider()
st.caption(
    "Educational portfolio project. The included dataset is synthetic "
    "and should not be used for real educational decisions."
)
