import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="HabitLens - Obesity Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark theme with proper card styling
st.markdown(
    """
<head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
</head>
<style>
    .stApp {
        background-color: #111827;
        color: #f9fafb;
    }
    
    /* Custom Card Styling */
    .custom-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .custom-card-header {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .custom-card-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .risk-low { background: #10b981; color: white; }
    .risk-moderate { background: #f59e0b; color: white; }
    .risk-high { background: #f97316; color: white; }
    .risk-very-high { background: #ef4444; color: white; }
    
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .recommendation-item {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .recommendation-number {
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 1.5rem;
        height: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    .factor-item {
        background: #fbbf24;
        color: #92400e;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        border-left: 6px solid #f59e0b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .success-item {
        background: #d1fae5;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        border-left: 4px solid #10b981;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: #d1d5db;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    /* Override Streamlit styles */
    .stSelectbox > div > div {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #f9fafb !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #f9fafb !important;
    }
    
    .stCheckbox > label {
        color: #f9fafb !important;
    }
    
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Hide Streamlit default elements */
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "assessment"
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "assessment_results" not in st.session_state:
    st.session_state.assessment_results = None


class ObesityAssessment:
    def __init__(self):
        self.total_steps = 4

    def calculate_bmi(self, height_cm, weight_kg):
        if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
            return None
        height_m = height_cm / 100
        return weight_kg / (height_m**2)

    def get_bmi_category(self, bmi):
        if bmi is None:
            return "N/A"
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        elif bmi < 35:
            return "Obesity Class I"
        elif bmi < 40:
            return "Obesity Class II"
        else:
            return "Obesity Class III"

    def calculate_risk_score(self, data):
        height = data.get("height", 0)
        weight = data.get("weight", 0)

        if not height or not weight:
            return {
                "bmi": None,
                "bmi_category": "N/A",
                "risk_score": 0,
                "risk_level": "Unknown",
                "key_factors": [],
                "confidence": 0,
            }

        bmi = self.calculate_bmi(height, weight)
        bmi_category = self.get_bmi_category(bmi)

        # Base risk from BMI
        if bmi is None:
            base_risk = 0
        elif bmi < 18.5:
            base_risk = 10
        elif bmi < 25:
            base_risk = 20
        elif bmi < 30:
            base_risk = 60
        elif bmi < 35:
            base_risk = 80
        elif bmi < 40:
            base_risk = 90
        else:
            base_risk = 95

        risk_score = base_risk
        key_factors = []

        # Add risk factors
        if data.get("smoker", False):
            risk_score += 5
            key_factors.append("Smoking habit: This can increase your risk of obesity")

        if data.get("family_history", False):
            risk_score += 10
            key_factors.append(
                "Family history of obesity: This can significantly increase your risk of obesity."
            )

        if data.get("eat_between_meals", False):
            risk_score += 8
            key_factors.append(
                "Frequent snacking between meals: This habit can lead to increased caloric intake and weight gain."
            )

        if data.get("high_calorie_food", False):
            risk_score += 12
            key_factors.append(
                "High-calorie food consumption: Regular intake of high-calorie foods can lead to weight gain and obesity."
            )

        if data.get("physical_activity") == "Never":
            risk_score += 15
            key_factors.append(
                "Sedentary lifestyle: Your lack of physical activity increases your risk of obesity in the future."
            )

        if data.get("technology_use", 0) > 6:
            risk_score += 5
            key_factors.append(
                "Excessive screen time: Spending too much time on screens can contribute to a sedentary lifestyle and weight gain."
            )

        if data.get("alcohol_consumption") in ["Frequently", "Always"]:
            risk_score += 7
            key_factors.append(
                "High alcohol consumption: Excessive alcohol intake can contribute to weight gain and obesity."
            )

        risk_score = min(risk_score, 100)

        # Determine risk level with 7 stages
        if risk_score < 15:
            risk_level = "Very Low Risk"
        elif risk_score < 30:
            risk_level = "Low Risk"
        elif risk_score < 45:
            risk_level = "Low-Moderate Risk"
        elif risk_score < 60:
            risk_level = "Moderate Risk"
        elif risk_score < 75:
            risk_level = "Moderate-High Risk"
        elif risk_score < 90:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"

        return {
            "bmi": round(bmi, 1) if bmi else None,
            "bmi_category": bmi_category,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "key_factors": key_factors,
            "confidence": 85 + np.random.random() * 10,
        }

    def generate_recommendations(self, data, risk_score, key_factors):
        recommendations = []

        if data.get("physical_activity") in ["Never", "1-2 times per week"]:
            recommendations.append(
                "Increase physical activity to at least 150 minutes of moderate exercise per week"
            )

        if data.get("eat_between_meals", False):
            recommendations.append(
                "Reduce snacking between meals and focus on balanced main meals"
            )

        if data.get("high_calorie_food", False):
            recommendations.append(
                "Replace high-calorie processed foods with nutrient-dense whole foods"
            )

        if data.get("water_consumption", 0) < 6:
            recommendations.append(
                "Increase daily water intake to at least 8 glasses per day"
            )

        if data.get("vegetable_consumption", 0) < 5:
            recommendations.append(
                "Aim for 5-7 servings of vegetables and fruits daily"
            )

        if data.get("smoker", False):
            recommendations.append(
                "Consider smoking cessation programs to improve overall health"
            )

        if not data.get("monitor_calories", False):
            recommendations.append(
                "Start tracking daily caloric intake to maintain awareness of eating patterns"
            )

        if data.get("technology_use", 0) > 6:
            recommendations.append(
                "Reduce screen time and incorporate more physical activities"
            )

        if risk_score > 70:
            recommendations.append(
                "Schedule regular follow-up appointments for monitoring progress"
            )
            recommendations.append(
                "Consider consultation with a registered dietitian for personalized meal planning"
            )

        return recommendations


def get_discrete_risk_level(score):
    if score <= 14:
        return 1, "Insufficient Weight"
    elif score <= 29:
        return 2, "Normal Weight"
    elif score <= 44:
        return 3, "Overweight Level I"
    elif score <= 59:
        return 4, "Overweight Level II"
    elif score <= 74:
        return 5, "Obesity Type I"
    elif score <= 89:
        return 6, "Obesity Type II"
    else:
        return 7, "Obesity Type III"


def create_risk_gauge(risk_score):
    """Create a Plotly gauge showing 7 discrete risk levels"""

    level, level_label = get_discrete_risk_level(risk_score)

    # Position needle at the center of each section (1-7 scale)
    # Each section spans 1 unit, so center is at level value
    needle_position = level

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=needle_position,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {
                    "range": [0.5, 7.5],
                    "tickvals": [1, 2, 3, 4, 5, 6, 7],
                    "ticktext": [
                        "Insufficient Weight",
                        "Normal Weight",
                        "Overweight Level I",
                        "Overweight Level II",
                        "Obesity Type I",
                        "Obesity Type II",
                        "Obesity Type III",
                    ],
                    "tickfont": {"color": "white", "size": 12},
                },
                "bar": {"color": "white", "thickness": 0.3},
                "steps": [
                    {
                        "range": [0.5, 1.5],
                        "color": "#fbbf24",
                    },  # Level 1: Insufficient Weight (purple - concerning)
                    {
                        "range": [1.5, 2.5],
                        "color": "#10b981",
                    },  # Level 2: Normal Weight (green - healthy)
                    {
                        "range": [2.5, 3.5],
                        "color": "#fbbf24",
                    },  # Level 3: Overweight Level I (yellow - caution)
                    {
                        "range": [3.5, 4.5],
                        "color": "#f59e0b",
                    },  # Level 4: Overweight Level II (amber - warning)
                    {
                        "range": [4.5, 5.5],
                        "color": "#f97316",
                    },  # Level 5: Obesity Type I (orange - concerning)
                    {
                        "range": [5.5, 6.5],
                        "color": "#ef4444",
                    },  # Level 6: Obesity Type II (red - dangerous)
                    {
                        "range": [6.5, 7.5],
                        "color": "#b91c1c",
                    },  # Level 7: Obesity Type III (dark red - severe)
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": needle_position,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                text=f"<b>{level_label}</b>",
                x=0.5,
                y=0.1,
                xref="paper",
                yref="paper",
                font=dict(size=30, color="white"),
                showarrow=False,
                align="center",
            ),
            dict(
                text=f"You might be at this level in the near future",
                x=0.5,
                y=0,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="white"),
                showarrow=False,
                align="center",
            ),
        ],
    )

    return fig


def render_assessment_page():
    assessment = ObesityAssessment()

    # Header
    st.markdown(
        """
            # :material/routine: HabitLens - Obesity Risk Assessment
            ##### Comprehensive patient evaluation for nutritional counseling
        """
    )

    # Progress indicator
    progress_value = st.session_state.current_step / assessment.total_steps

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(
            f"**Step {st.session_state.current_step} of {assessment.total_steps}**"
        )
    with col2:
        st.markdown(f"**{int(progress_value * 100)}% Complete**")

    st.progress(progress_value)

    # Step content
    if st.session_state.current_step == 1:
        render_step_1()
    elif st.session_state.current_step == 2:
        render_step_2()
    elif st.session_state.current_step == 3:
        render_step_3()
    elif st.session_state.current_step == 4:
        render_step_4()

    (
        col1,
        col2,
        col3,
    ) = st.columns([1, 2, 1])

    with col1:
        if st.session_state.current_step > 1:
            if st.button("← Previous", key="prev_btn"):
                st.session_state.current_step -= 1
                st.rerun()

    with col3:
        if st.session_state.current_step < assessment.total_steps:
            if st.button("Next →", key="next_btn"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button(
                ":material/rocket_launch: Generate Assessment", key="generate_btn"
            ):
                generate_assessment()


def render_step_1():
    st.markdown(
        """
            ### :material/account_circle: Personal Information
            Please provide your personal information. The data will be used to assess your obesity risk and generate personalized recommendations.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            key="age_input",
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100,
            max_value=250,
            value=170,
            key="height_input",
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="gender_input",
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30,
            max_value=300,
            value=70,
            key="weight_input",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {"age": age, "gender": gender, "height": height, "weight": weight}
    )


def render_step_2():

    st.markdown(
        """
            ### :material/directions_bike: Lifestyle & Habits
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        physical_activity = st.selectbox(
            "Physical Activity Frequency",
            ["Never", "1-2 times per week", "2-4 times per week", "4+ times per week"],
            key="physical_activity_input",
        )

        transportation = st.selectbox(
            "Primary Transportation Mode",
            ["Walking", "Bicycle", "Public Transportation", "Car"],
            index=3,  # Default to "Car"
            key="transportation_input",
        )

    with col2:
        technology_use = st.number_input(
            "Technology Use (screen time hours/day)",
            min_value=0,
            max_value=24,
            value=4,
            key="technology_use_input",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "physical_activity": physical_activity,
            "transportation": transportation,
            "technology_use": technology_use,
        }
    )


def render_step_3():
    st.markdown(
        """  
            ### :material/restaurant_menu: Dietary Habits
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        meals_per_day = st.number_input(
            "Number of Main Meals per Day",
            min_value=1,
            max_value=10,
            value=3,
            key="meals_per_day_input",
        )

        water_consumption = st.number_input(
            "Water Consumption (glasses/day)",
            min_value=0,
            max_value=20,
            step=1,
            value=6,
            key="water_consumption_input",
        )

    with col2:
        vegetable_consumption = st.number_input(
            "Vegetable Consumption (servings/day)",
            min_value=0,
            max_value=20,
            value=3,
            key="vegetable_consumption_input",
        )

        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            ["Never", "Sometimes", "Frequently", "Always"],
            key="alcohol_consumption_input",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "meals_per_day": meals_per_day,
            "water_consumption": water_consumption,
            "vegetable_consumption": vegetable_consumption,
            "alcohol_consumption": alcohol_consumption,
        }
    )


def render_step_4():
    st.markdown(
        """            
            ### :material/health_and_safety: Health & Lifestyle Factors
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        smoker = st.checkbox(
            "Do you smoke?",
            key="smoker_input",
        )

        family_history = st.checkbox(
            "Family history of obesity?",
            key="family_history_input",
        )

        eat_between_meals = st.checkbox(
            "Do you eat food between meals?",
            key="eat_between_meals_input",
        )

    with col2:
        high_calorie_food = st.checkbox(
            "Do you frequently consume high-calorie foods?",
            key="high_calorie_food_input",
        )

        monitor_calories = st.checkbox(
            "Do you monitor your calorie intake?",
            key="monitor_calories_input",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "smoker": smoker,
            "family_history": family_history,
            "eat_between_meals": eat_between_meals,
            "high_calorie_food": high_calorie_food,
            "monitor_calories": monitor_calories,
        }
    )


def generate_assessment():
    assessment = ObesityAssessment()
    results = assessment.calculate_risk_score(st.session_state.patient_data)
    recommendations = assessment.generate_recommendations(
        st.session_state.patient_data, results["risk_score"], results["key_factors"]
    )
    results["recommendations"] = recommendations
    st.session_state.assessment_results = results
    st.session_state.current_page = "results"
    st.rerun()


def get_risk_badge_class(risk_level):
    """Get CSS class for risk level badge"""
    if "Very Low" in risk_level or "Low" in risk_level:
        return "risk-low"
    elif "Moderate" in risk_level:
        return "risk-moderate"
    elif "High" in risk_level:
        return "risk-high"
    else:
        return "risk-very-high"


def render_results_page():

    # Header
    st.markdown(
        """
        # :material/analytics: Assessment Results
        #### Your AI-powered obesity risk analysis and recommendations
        """,
    )

    results = st.session_state.assessment_results
    data = st.session_state.patient_data

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ## :material/balance: Future Obesity Risk Assessment
        """
        )

        # Plotly gauge chart (outside HTML but visually inside)
        gauge_fig = create_risk_gauge(results["risk_score"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        # Add spacing to align with Next Steps
        st.markdown("<br><br>", unsafe_allow_html=True)

        # BMI Information - positioned to align with Next Steps
        col_bmi1, col_bmi2 = st.columns(2)
        with col_bmi1:
            bmi_display = results["bmi"] if results["bmi"] else "—"
            st.markdown(
                f"""
            <div class="metric-card">
                <p style="color: #9ca3af; font-size: 0.875rem;">Current BMI</p>
                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">{bmi_display}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_bmi2:
            bmi_category = results["bmi_category"] if results["bmi_category"] else "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <p style="color: #9ca3af; font-size: 0.875rem;">Current BMI Category</p>
                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">{bmi_category}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        # Patient Info Card
        st.markdown(
            """
            <div class="custom-card" style="align-items: center; justify-content: center; flex-direction: column;">
                <div class="custom-card-title" style="align-items: center; justify-content: center; display: flex;">
                    <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">body_system</span>
                    <h3 style="font-weight: 600; margin: 0; text-align: center;">Patient Info</h3>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.95rem; color: #9ca3af;">Age</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{age} yrs</div>
                    </div>
                        <div style="text-align: center;">
                        <div style="font-size: 0.95rem; color: #9ca3af;">Gender</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">{gender}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.95rem; color: #9ca3af;">Height</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{height} cm</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.95rem; color: #9ca3af;">Weight</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{weight} kg</div>
                    </div>
                </div>
            </div>
            """.format(
                age=data["age"],
                gender=data["gender"],
                height=data["height"],
                weight=data["weight"],
            ),
            unsafe_allow_html=True,
        )

        # Model Confidence Card
        confidence = int(results["confidence"] / 10)
        st.markdown(
            f"""
        <div class="custom-card">
            <div class="custom-card-title" style="align-items: center; justify-content: center; display: flex;">
                    <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">bolt</span>
                    <h3 style="font-weight: 600; margin: 0; text-align: center;">Model Confidence</h3>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">{confidence}/10</div>
                <p style="color: #9ca3af; font-size: 0.8rem;">(How sure the model is of its answer, or how likely it is that the model is correct)</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Actions Card
        st.markdown(
            """
                <div class="custom-card-title" style="align-items: center; justify-content: center; display: flex;">
                    <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">arrow_circle_right</span>
                    <h3 style="font-weight: 600; margin: 0; text-align: center; margin-top: 0.5rem">Next Steps</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Generate Patient Report", key="report_btn", use_container_width=True
            ):
                st.session_state.current_page = "report"
                st.rerun()

        with col2:
            if st.button(
                ":material/autorenew: New Assessment",
                key="new_assessment_btn",
                use_container_width=True,
            ):
                st.session_state.current_page = "assessment"
                st.session_state.current_step = 1
                st.session_state.patient_data = {}
                st.session_state.assessment_results = None
                st.rerun()

    # Risk Factors Card
    st.markdown(
        """
            ## :material/emergency_home: Risk Factors
        """
    )

    if results["key_factors"]:
        for factor in results["key_factors"]:
            st.markdown(
                f"""
            <div class="recommendation-item">
                <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">person_alert</span> 
                <p style="font-size: 1rem; margin: 0; margin-top: 0.25rem">{factor}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
        <div class="success-item">
            ✅ No significant risk factors identified
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recommendations Card
    st.markdown(
        """
            ## :material/lightbulb: Recommendations
        """
    )

    for i, recommendation in enumerate(results["recommendations"], 1):
        st.markdown(
            f"""
        <div class="recommendation-item">
                <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">recommend</span> 
                <p style="font-size: 1rem; margin: 0; margin-top: 0.25rem">{recommendation}</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Recommendations Card
    st.markdown(
        """
            ## :material/neurology: Model Explanations
        """
    )

    st.info(
        """
    This assessment uses a comprehensive machine learning model that analyzes 16 different factors including
    demographic information, lifestyle habits, dietary patterns, and behavioral indicators. The model has
    been trained on extensive nutritional and health datasets to provide accurate obesity risk predictions.
    The risk score is calculated by weighing each factor according to its clinical significance in obesity
    development.
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div style="font-size: 1.5rem; color: #10b981; margin-bottom: 0.5rem;">
                <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">sprint</span>
            </div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Lifestyle Factors</div>
            <div style="color: #9ca3af; font-size: 0.875rem;">Physical activity, transportation, technology use</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div style="font-size: 1.5rem; color: #f97316; margin-bottom: 0.5rem;"><span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">dining</span></div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Dietary Patterns</div>
            <div style="color: #9ca3af; font-size: 0.875rem;">Meal frequency, nutrition quality, hydration</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div style="font-size: 1.5rem; color: #ef4444; margin-bottom: 0.5rem;"><span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">favorite</span></div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Health Indicators</div>
            <div style="color: #9ca3af; font-size: 0.875rem;">BMI, family history, behavioral habits</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_report_page():
    results = st.session_state.assessment_results
    data = st.session_state.patient_data
    report_date = datetime.now().strftime("%B %d, %Y")

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        if st.button(
            ":material/arrow_back: Back to Results",
            key="back_to_results_top",
        ):
            st.session_state.current_page = "results"
            st.rerun()

    # Header with print button
    st.markdown(
        f"""
    <div class="custom-card-header">
        <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 0.5rem;"><span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">assignment</span> Patient Assessment Report</h1>
        <p style="color: #9ca3af;">Generated on {report_date}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Report Content
    st.markdown(
        """
        ## :material/account_circle: Patient Information
    """,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            value=data["age"], label="Age", delta=None, help="Patient's age in years"
        )
    with col2:
        st.metric(
            value=data["gender"], label="Gender", delta=None, help="Patient's gender"
        )
    with col3:
        st.metric(
            value=data["height"],
            label="Height",
            delta=None,
            help="Patient's height in cm",
        )
    with col4:
        st.metric(
            value=data["weight"],
            label="Weight",
            delta=None,
            help="Patient's weight in kg",
        )

    st.markdown("---")

    # Assessment Results
    st.markdown(
        """
        ## :material/analytics: Assessment Results
    """,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            value=results["risk_level"],
            label="Risk Level",
            delta=None,
            help="Overall risk level based on assessment",
        )

    _, level_label = get_discrete_risk_level(results["risk_score"])

    with col2:
        st.metric(
            value=level_label,
            label="Projected Weight Level",
            delta=None,
            help="Projected weight level based on risk score",
        )

    with col3:
        st.metric(
            value=results["bmi_category"],
            label="Current BMI Category",
            delta=None,
            help="Current BMI category based on height and weight",
        )

    st.markdown("---")

    # Risk Factors
    st.markdown(
        """
        ## :material/emergency_home: Risk Factors
    """,
    )

    if results["key_factors"]:
        for i, factor in enumerate(results["key_factors"], 1):
            st.markdown(f"#### {i}. {factor}")
    else:
        st.success(
            "#### :material/check_circle: No significant risk factors identified"
        )

    st.markdown("---")

    # Recommendations
    st.markdown("## :material/lightbulb: Personalized Recommendations")
    for i, recommendation in enumerate(results["recommendations"], 1):
        st.markdown(f"#### **{i}.** {recommendation}")

    st.markdown("---")

    # Lifestyle Analysis
    st.markdown("## :material/directions_run: Lifestyle Habits")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### **Physical Activity**")
        st.metric(
            label="Physical Activity Frequency",
            value=data.get("physical_activity", "Not specified"),
            help="How often the patient engages in physical activity",
        )
        st.metric(
            label="Primary Transportation",
            value=data.get("transportation", "Not specified"),
            help="Main mode of transportation",
        )
        st.metric(
            label="Screen Time (hrs/day)",
            value=data.get("technology_use", 0),
            help="Average daily technology/screen usage",
        )

    with col2:
        st.markdown("### **Dietary Habits**")
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="Meals/day",
                value=data.get("meals_per_day", "Not specified"),
            )
            st.metric(
                label="Water intake",
                value=data.get("water_consumption", 0),
                help="Average daily water consumption",
            )
        with col4:
            st.metric(
                label="Vegetables",
                value=data.get("vegetable_consumption", 0),
                help="Average daily vegetable servings",
            )
            st.metric(
                label="Alcohol",
                value=data.get("alcohol_consumption", "Not specified"),
                help="Average daily alcohol consumption",
            )

    st.markdown("---")

    # Follow-up Plan
    st.markdown("## :material/calendar_month: Follow-up Plan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### **Short-term Goals (1-3 months):**")
        st.metric(label="Key Recommendations", value="Implement 2-3")
        st.metric(label="Track Progress", value="Daily food & activity log")
        st.metric(label="Follow-up", value="Schedule appointment")

    with col2:
        st.markdown("### **Long-term Goals (3-12 months):**")
        st.metric(label="Target Weight", value="Achieve healthy range")
        st.metric(label="Lifestyle Habits", value="Sustainable routines")
        st.metric(label="Health Monitoring", value="Regular check-ups")

    # Next appointment
    st.info(
        "**Next Appointment:** Schedule a follow-up consultation in 4-6 weeks to monitor progress and adjust recommendations as needed."
    )

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style="text-align: center; color: #6b7280; font-size: 0.8rem;">
        <p>This report was generated using AI-powered nutritional assessment tools.</p>
        <p>For questions or concerns, please consult with your healthcare provider.</p>
        <p>Report ID: ORA-{datetime.now().strftime('%Y%m%d%H%M')} | Generated: {report_date}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation
    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        if st.button(
            ":material/arrow_back: Back to Results",
            key="back_to_results",
        ):
            st.session_state.current_page = "results"
            st.rerun()

    with col2:
        if st.button(":material/refresh: New Assessment", key="new_from_report"):
            st.session_state.current_page = "assessment"
            st.session_state.current_step = 1
            st.session_state.patient_data = {}
            st.session_state.assessment_results = None
            st.rerun()


# Main app logic
def main():
    if st.session_state.current_page == "assessment":
        render_assessment_page()
    elif st.session_state.current_page == "results":
        render_results_page()
    elif st.session_state.current_page == "report":
        render_report_page()


if __name__ == "__main__":
    main()
