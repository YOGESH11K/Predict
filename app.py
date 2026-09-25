# ==========================================
# AI STUDENT PERFORMANCE PREDICTOR
# Class 9 Python Project
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 AI Student Performance Predictor")
st.subheader("🤖 Class 9 Machine Learning Project")

st.write(
    "This AI model predicts a student's final marks "
    "using study hours, attendance, and previous marks."
)

st.divider()


# ==========================================
# STUDENT DATA
# ==========================================

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Previous_Marks": [40, 45, 50, 58, 65, 72, 80, 88],
    "Final_Marks": [42, 47, 53, 60, 67, 75, 82, 90]
}

df = pd.DataFrame(data)


# ==========================================
# TRAIN AI MODEL
# ==========================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks"
    ]
]

y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎓 Student Predictor")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "🏠 Prediction",
        "📊 Student Data",
        "📈 Analysis"
    ]
)


# ==========================================
# PREDICTION PAGE
# ==========================================

if page == "🏠 Prediction":

    st.header("🤖 Predict Student Performance")

    st.write(
        "Enter the student's information below."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.number_input(
            "📚 Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

    with col2:
        attendance = st.number_input(
            "🏫 Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col3:
        previous_marks = st.number_input(
            "📝 Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

    st.write("")

    if st.button(
        "🔮 Predict Final Marks",
        use_container_width=True
    ):

        new_student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks]
        })

        prediction = model.predict(new_student)

        predicted_marks = prediction[0]

        # Keep marks between 0 and 100
        predicted_marks = max(
            0,
            min(100, predicted_marks)
        )

        st.divider()

        st.subheader("🎯 AI Prediction")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Predicted Marks",
                f"{predicted_marks:.2f} / 100"
            )

        with result_col2:
            if predicted_marks >= 40:
                st.success("PASS ✅")
            else:
                st.error("NEED MORE PRACTICE 📚")

        with result_col3:

            if predicted_marks >= 80:
                level = "Excellent 🌟"
            elif predicted_marks >= 60:
                level = "Good 👍"
            elif predicted_marks >= 40:
                level = "Needs Improvement 💪"
            else:
                level = "Beginner 📖"

            st.metric(
                "Performance Level",
                level
            )

        st.divider()

        # ==========================================
        # PROGRESS BAR
        # ==========================================

        st.write("### 📊 Performance Score")

        st.progress(int(predicted_marks))

        # ==========================================
        # AI FEEDBACK
        # ==========================================

        st.write("### 🤖 AI Feedback")

        if predicted_marks >= 80:

            st.success(
                "🌟 Excellent performance! "
                "Keep up the great work!"
            )

        elif predicted_marks >= 60:

            st.info(
                "👍 Good performance. "
                "Keep improving!"
            )

        elif predicted_marks >= 40:

            st.warning(
                "💪 You can improve with "
                "more practice."
            )

        else:

            st.error(
                "📖 Focus more on your studies "
                "and practice regularly."
            )


# ==========================================
# STUDENT DATA PAGE
# ==========================================

elif page == "📊 Student Data":

    st.header("📊 Student Training Data")

    st.write(
        "The AI model is trained using the following "
        "sample student data:"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Study Hours",
            f"{df['Study_Hours'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Average Attendance",
            f"{df['Attendance'].mean():.2f}%"
        )

    with col3:
        st.metric(
            "Average Marks",
            f"{df['Final_Marks'].mean():.2f}"
        )


# ==========================================
# ANALYSIS PAGE
# ==========================================

elif page == "📈 Analysis":

    st.header("📈 Student Performance Analysis")

    st.write(
        "This graph shows the relationship between "
        "study hours and final marks."
    )

    fig, ax = plt.subplots()

    ax.scatter(
        df["Study_Hours"],
        df["Final_Marks"]
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Marks")

    ax.set_title(
        "Study Hours vs Final Marks"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("📊 Basic Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Study Hours",
            f"{df['Study_Hours'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Average Attendance",
            f"{df['Attendance'].mean():.2f}%"
        )

    with col3:
        st.metric(
            "Average Final Marks",
            f"{df['Final_Marks'].mean():.2f}"
        )

    st.divider()

    st.subheader("🤖 AI Model Status")

    st.success(
        "✅ Linear Regression model trained successfully!"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "🎓 AI Student Performance Predictor | "
    "Class 9 Python & Machine Learning Project"
)