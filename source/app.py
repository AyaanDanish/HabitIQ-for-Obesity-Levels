import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import joblib
import os


# Adjusted Importance Model Class (needed for loading pickled adjusted models)
class AdjustedImportanceModel:
    def __init__(self, model, feature_names, adjusted_importance):
        self.model = model
        self.feature_names = feature_names
        self.adjusted_importance = adjusted_importance
        self.original_importance = model.feature_importances_

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_adjusted_importance(self):
        return self.adjusted_importance

    def get_original_importance(self):
        return self.original_importance

    @property
    def classes_(self):
        return self.model.classes_

    @property
    def feature_importances_(self):
        # Return adjusted importance by default
        return self.adjusted_importance


# Get the directory of this script to build absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# Load the trained model (with adjusted importance if available)
@st.cache_resource
def load_model():
    # Try to load the adjusted model first
    adjusted_model_path = os.path.join(
        PROJECT_ROOT, "model", "obesity_model_adjusted.pkl"
    )
    original_model_path = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")

    if os.path.exists(adjusted_model_path):
        try:
            adjusted_model = joblib.load(adjusted_model_path)
            st.success(
                "✅ **Loaded adjusted model** - Using lifestyle-focused feature importance"
            )
            return adjusted_model
        except Exception as e:
            st.warning(
                f"Failed to load adjusted model: {e}. Falling back to original model."
            )

    if os.path.exists(original_model_path):
        original_model = joblib.load(original_model_path)
        st.info("ℹ️ **Loaded original model** - Using standard feature importance")
        return original_model
    else:
        st.error(
            f"No model files found. Please ensure either 'obesity_model_adjusted.pkl' or 'random_forest_model.pkl' exists in the model directory."
        )
        return None


# Load the training data to get feature names
@st.cache_data
def get_feature_names():
    """Get the exact feature names used during training"""
    try:
        data_path = os.path.join(PROJECT_ROOT, "data", "ObesityData.csv")
        df = pd.read_csv(data_path)

        # Apply same preprocessing as train.py
        binary_cols = ["FAVC", "SCC", "SMOKE", "family_history_with_overweight"]
        for col in binary_cols:
            unique_vals = sorted(df[col].unique())
            mapping = {unique_vals[0]: False, unique_vals[1]: True}
            df[col] = df[col].map(mapping)

        # One-hot encode to get feature names
        X = pd.get_dummies(df.drop("NObeyesdad", axis=1), drop_first=True)
        return list(X.columns), df["NObeyesdad"].unique()
    except FileNotFoundError:
        st.warning(
            f"ObesityData.csv not found at {data_path}. Using expected feature names from model training."
        )
        # Fallback feature names based on typical model structure
        fallback_features = [
            "Age",
            "Height",
            "Weight",
            "FAVC",
            "FCVC",
            "NCP",
            "SCC",
            "SMOKE",
            "CH2O",
            "family_history_with_overweight",
            "FAF",
            "TUE",
            "Gender_Male",
            "CALC_Frequently",
            "CALC_Sometimes",
            "CAEC_Always",
            "CAEC_Frequently",
            "CAEC_Sometimes",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Walking",
        ]
        fallback_classes = [
            "Insufficient_Weight",
            "Normal_Weight",
            "Overweight_Level_I",
            "Overweight_Level_II",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III",
        ]
        return fallback_features, fallback_classes
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        # Return fallback feature names
        fallback_features = [
            "Age",
            "Height",
            "Weight",
            "FAVC",
            "FCVC",
            "NCP",
            "SCC",
            "SMOKE",
            "CH2O",
            "family_history_with_overweight",
            "FAF",
            "TUE",
            "Gender_Male",
            "CALC_Frequently",
            "CALC_Sometimes",
            "CAEC_Always",
            "CAEC_Frequently",
            "CAEC_Sometimes",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Walking",
        ]
        fallback_classes = [
            "Insufficient_Weight",
            "Normal_Weight",
            "Overweight_Level_I",
            "Overweight_Level_II",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III",
        ]
        return fallback_features, fallback_classes


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
    
    .risk-factor-item {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-left: 4px solid #ef4444;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(239, 68, 68, 0.2);
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
    
    /* Navigation button alignment - Cloud-compatible selectors */
    
    /* Left button alignment */
    div[data-testid="column"]:has(.nav-button-left) .stButton {
        display: flex !important;
        justify-content: flex-start !important;
    }
    
    .nav-button-left {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
    }
    
    /* Right button alignment - Multiple approaches for cloud compatibility */
    div[data-testid="column"]:has(.nav-button-right) {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }
    
    div[data-testid="column"]:has(.nav-button-right) .stButton {
        margin-left: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
    }
    
    .nav-button-right {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-end !important;
        text-align: right !important;
    }
    
    /* Target all possible button structures within nav-button-right */
    .nav-button-right .stButton {
        margin-left: auto !important;
        width: auto !important;
        display: inline-flex !important;
        justify-content: flex-end !important;
    }
    
    .nav-button-right .stButton > div {
        margin-left: auto !important;
        width: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
    }
    
    .nav-button-right .stButton > div > div {
        margin-left: auto !important;
        width: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
    }
    
    .nav-button-right .stButton button {
        margin-left: auto !important;
    }
    
    /* More specific targeting for cloud deployment */
    .nav-button-right button[kind="secondary"] {
        margin-left: auto !important;
    }
    
    .nav-button-right button[kind="primary"] {
        margin-left: auto !important;
    }
    
    .nav-button-right [data-testid*="button"] {
        margin-left: auto !important;
    }
    
    .nav-button-right [class*="stButton"] {
        margin-left: auto !important;
        justify-content: flex-end !important;
    }
    
    /* Force right alignment with CSS grid as backup */
    .nav-button-right {
        display: grid !important;
        grid-template-columns: 1fr auto !important;
        width: 100% !important;
    }
    
    .nav-button-right > * {
        grid-column: 2 !important;
    }
    
    /* Hide Streamlit default elements */
</style>

<script>
// JavaScript fallback for button alignment in cloud deployment
document.addEventListener('DOMContentLoaded', function() {
    function alignNavigationButtons() {
        // Find all nav-button-right containers
        const rightNavButtons = document.querySelectorAll('.nav-button-right');
        
        rightNavButtons.forEach(container => {
            // Apply styles directly via JavaScript
            container.style.width = '100%';
            container.style.display = 'flex';
            container.style.justifyContent = 'flex-end';
            container.style.textAlign = 'right';
            
            // Find the button within this container
            const button = container.querySelector('button');
            if (button) {
                button.style.marginLeft = 'auto';
            }
            
            // Also target any stButton divs
            const stButtons = container.querySelectorAll('[class*="stButton"], .stButton');
            stButtons.forEach(stButton => {
                stButton.style.marginLeft = 'auto';
                stButton.style.display = 'flex';
                stButton.style.justifyContent = 'flex-end';
            });
        });
        
        // Find all nav-button-left containers
        const leftNavButtons = document.querySelectorAll('.nav-button-left');
        leftNavButtons.forEach(container => {
            container.style.width = '100%';
            container.style.display = 'flex';
            container.style.justifyContent = 'flex-start';
            container.style.textAlign = 'left';
        });
    }
    
    // Run immediately
    alignNavigationButtons();
    
    // Run again after a short delay to catch dynamically loaded content
    setTimeout(alignNavigationButtons, 500);
    setTimeout(alignNavigationButtons, 1000);
    setTimeout(alignNavigationButtons, 2000);
    
    // Create a MutationObserver to watch for DOM changes
    const observer = new MutationObserver(function(mutations) {
        let shouldAlign = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' || mutation.type === 'attributes') {
                // Check if any nav-button elements were added or modified
                const target = mutation.target;
                if (target.classList && (target.classList.contains('nav-button-right') || 
                    target.classList.contains('nav-button-left') ||
                    target.querySelector && (target.querySelector('.nav-button-right') || target.querySelector('.nav-button-left')))) {
                    shouldAlign = true;
                }
            }
        });
        
        if (shouldAlign) {
            setTimeout(alignNavigationButtons, 100);
        }
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style']
    });
});
</script>
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
        self.model = load_model()
        self.feature_names, self.class_labels = get_feature_names()

    def calculate_bmi(self, height_m, weight_kg):
        if not height_m or not weight_kg or height_m <= 0 or weight_kg <= 0:
            return None
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

    def prepare_input_data(self, data):
        """Convert Streamlit form data to model input format"""
        if not self.model or not self.feature_names:
            return None

        # Create a DataFrame with all possible features, initialized to 0/False
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)

        # Set the numeric features
        input_df["Age"] = data.get("age", 30)
        input_df["Height"] = data.get("height", 1.7)  # Already in meters from app
        input_df["Weight"] = data.get("weight", 70)
        input_df["FCVC"] = data.get("vegetable_consumption", 2.0)  # 1-3 scale
        input_df["NCP"] = data.get("meals_per_day", 2.0)  # 1-3 scale
        input_df["CH2O"] = data.get("water_consumption", 2.0)  # 1-3 scale
        input_df["FAF"] = data.get("physical_activity", 1.0)  # 0-3 scale
        input_df["TUE"] = data.get("technology_use", 1.0)  # 0-2 scale

        # Set boolean features
        input_df["FAVC"] = data.get("high_calorie_food", False)
        input_df["SCC"] = data.get("monitor_calories", False)
        input_df["SMOKE"] = data.get("smoker", False)
        input_df["family_history_with_overweight"] = data.get("family_history", False)

        # Handle categorical features with one-hot encoding
        # Gender
        if data.get("gender") == "Male" and "Gender_Male" in self.feature_names:
            input_df["Gender_Male"] = True

        # CALC (alcohol consumption)
        alcohol = data.get("alcohol_consumption", "no")
        if alcohol != "no":
            if alcohol == "Sometimes" and "CALC_Sometimes" in self.feature_names:
                input_df["CALC_Sometimes"] = True
            elif alcohol == "Frequently" and "CALC_Frequently" in self.feature_names:
                input_df["CALC_Frequently"] = True

        # CAEC (eat between meals)
        caec = data.get("eat_between_meals", "no")
        if caec != "no":
            if caec == "Sometimes" and "CAEC_Sometimes" in self.feature_names:
                input_df["CAEC_Sometimes"] = True
            elif caec == "Frequently" and "CAEC_Frequently" in self.feature_names:
                input_df["CAEC_Frequently"] = True
            elif caec == "Always" and "CAEC_Always" in self.feature_names:
                input_df["CAEC_Always"] = True

        # MTRANS (transportation)
        transport = data.get("transportation", "Public_Transportation")
        transport_cols = {
            "Automobile": "MTRANS_Automobile",
            "Bike": "MTRANS_Bike",
            "Motorbike": "MTRANS_Motorbike",
            "Walking": "MTRANS_Walking",
        }
        if (
            transport in transport_cols
            and transport_cols[transport] in self.feature_names
        ):
            input_df[transport_cols[transport]] = True

        return input_df

    def calculate_risk_score(self, data):
        if not self.model:
            return self._fallback_risk_calculation(data)

        # Prepare input data
        input_data = self.prepare_input_data(data)
        if input_data is None:
            return self._fallback_risk_calculation(data)

        try:  # Get prediction and probabilities
            # Handle both original model and adjusted model wrapper
            if hasattr(self.model, "predict"):
                prediction = self.model.predict(input_data)[0]
                probabilities = self.model.predict_proba(input_data)[0]
            else:
                # Fallback if model structure is unexpected
                st.error("Model prediction methods not available")
                return self._fallback_risk_calculation(data)

            # Calculate BMI
            height = data.get("height", 1.7)
            weight = data.get("weight", 70)
            bmi = self.calculate_bmi(height, weight)
            bmi_category = self.get_bmi_category(bmi)

            # Convert prediction to risk score (0-100)
            risk_score = self._prediction_to_risk_score(prediction, probabilities)

            # Get risk level
            risk_level = self._get_risk_level_from_prediction(prediction)

            # Generate key factors based on model feature importance
            key_factors = self._generate_key_factors(data, input_data)

            return {
                "bmi": round(bmi, 1) if bmi else None,
                "bmi_category": bmi_category,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "prediction": prediction,
                "probabilities": probabilities,
                "key_factors": key_factors,
                "confidence": max(probabilities) * 100,
            }

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return self._fallback_risk_calculation(data)

    def _prediction_to_risk_score(self, prediction, probabilities):
        """Convert model prediction to 0-100 risk score"""
        # Map obesity classes to risk scores
        risk_mapping = {
            "Insufficient_Weight": 15,
            "Normal_Weight": 25,
            "Overweight_Level_I": 45,
            "Overweight_Level_II": 65,
            "Obesity_Type_I": 80,
            "Obesity_Type_II": 90,
            "Obesity_Type_III": 95,
        }

        base_score = risk_mapping.get(prediction, 50)
        # Adjust based on prediction confidence
        confidence = max(probabilities)
        adjusted_score = base_score * confidence + (50 * (1 - confidence))

        return min(int(adjusted_score), 100)

    def _get_risk_level_from_prediction(self, prediction):
        """Convert model prediction to risk level description"""
        risk_levels = {
            "Insufficient_Weight": "Low Risk",
            "Normal_Weight": "Very Low Risk",
            "Overweight_Level_I": "Low-Moderate Risk",
            "Overweight_Level_II": "Moderate Risk",
            "Obesity_Type_I": "Moderate-High Risk",
            "Obesity_Type_II": "High Risk",
            "Obesity_Type_III": "Very High Risk",
        }
        return risk_levels.get(prediction, "Moderate Risk")

    def _generate_key_factors(self, data, input_data):
        """Generate key risk factors based on model feature importance and predictions"""
        if not self.model or input_data is None:
            return self._fallback_key_factors(data)

        try:
            # Get feature importance from the trained model
            # Check if this is an adjusted model with custom importance
            if hasattr(self.model, "get_adjusted_importance"):
                feature_importance = dict(
                    zip(self.feature_names, self.model.get_adjusted_importance())
                )
                st.info(
                    "🎯 **Using adjusted feature importance** - Lifestyle factors prioritized"
                )
            else:
                feature_importance = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
                st.info(
                    "📊 **Using original feature importance** - Standard model weights"
                )

            # Calculate feature contributions for this specific patient
            patient_features = {}

            # Analyze each feature and its contribution to risk
            risk_factors = []

            # Check high-impact features based on model importance and patient values
            high_risk_features = []

            # Physical Activity (FAF) - High importance feature
            if "FAF" in feature_importance and feature_importance["FAF"] > 0.05:
                faf_value = input_data["FAF"].iloc[0]
                if faf_value <= 1.0:  # Low physical activity
                    impact_score = feature_importance["FAF"] * (
                        2.0 - faf_value
                    )  # Higher impact for lower activity
                    if faf_value == 0:
                        message = "You don't exercise regularly, which increases your risk of weight gain"
                    else:
                        message = "Your current activity level is quite low and could lead to weight gain over time"
                    high_risk_features.append(
                        {
                            "feature": "Physical Activity",
                            "impact": impact_score,
                            "message": message,
                            "severity": "high" if faf_value == 0 else "medium",
                        }
                    )

            # Family History - Genetic predisposition
            if (
                "family_history_with_overweight" in feature_importance
                and input_data["family_history_with_overweight"].iloc[0]
            ):
                impact_score = feature_importance["family_history_with_overweight"]
                high_risk_features.append(
                    {
                        "feature": "Family History",
                        "impact": impact_score,
                        "message": "Having family members with obesity increases your own risk due to shared genetics and lifestyle patterns",
                        "severity": "high",
                    }
                )

            # High Calorie Food Consumption (FAVC)
            if "FAVC" in feature_importance and input_data["FAVC"].iloc[0]:
                impact_score = feature_importance["FAVC"]
                high_risk_features.append(
                    {
                        "feature": "Diet Quality",
                        "impact": impact_score,
                        "message": "You frequently eat high-calorie foods, which can lead to weight gain if not balanced with physical activity",
                        "severity": "high",
                    }
                )

            # Smoking (SMOKE)
            if "SMOKE" in feature_importance and input_data["SMOKE"].iloc[0]:
                impact_score = feature_importance["SMOKE"]
                high_risk_features.append(
                    {
                        "feature": "Smoking",
                        "impact": impact_score,
                        "message": "Smoking can affect your metabolism and make it harder to maintain a healthy weight",
                        "severity": "high",
                    }
                )

            # Technology Use (TUE) - Screen time
            if "TUE" in feature_importance and feature_importance["TUE"] > 0.03:
                tue_value = input_data["TUE"].iloc[0]
                if tue_value >= 2.0:  # High screen time
                    impact_score = feature_importance["TUE"] * tue_value
                    high_risk_features.append(
                        {
                            "feature": "Screen Time",
                            "impact": impact_score,
                            "message": "You spend a lot of time on screens, which often means less physical activity and more sedentary behavior",
                            "severity": "medium",
                        }
                    )

            # Eating Between Meals (CAEC)
            caec_features = [f for f in self.feature_names if f.startswith("CAEC_")]
            for caec_feature in caec_features:
                if (
                    caec_feature in feature_importance
                    and input_data[caec_feature].iloc[0]
                ):
                    impact_score = feature_importance[caec_feature]
                    frequency = caec_feature.split("_")[1].lower()
                    if frequency == "sometimes":
                        message = "You sometimes snack between meals, which can add extra calories to your daily intake"
                    elif frequency == "frequently":
                        message = "You frequently snack between meals, which significantly increases your daily calorie consumption"
                    else:  # Always
                        message = "You regularly eat between meals, which can lead to consuming more calories than your body needs"

                    high_risk_features.append(
                        {
                            "feature": "Snacking Patterns",
                            "impact": impact_score,
                            "message": message,
                            "severity": (
                                "medium" if frequency == "sometimes" else "high"
                            ),
                        }
                    )
                    break

            # Water Consumption (CH2O) - Low water intake
            if "CH2O" in feature_importance:
                ch2o_value = input_data["CH2O"].iloc[0]
                if ch2o_value < 2.0:  # Low water consumption
                    impact_score = feature_importance["CH2O"] * (2.0 - ch2o_value)
                    high_risk_features.append(
                        {
                            "feature": "Hydration",
                            "impact": impact_score,
                            "message": "You don't drink enough water, which can slow down your metabolism and affect your body's ability to process food efficiently",
                            "severity": "low",
                        }
                    )

            # Vegetable Consumption (FCVC) - Low intake
            if "FCVC" in feature_importance:
                fcvc_value = input_data["FCVC"].iloc[0]
                if fcvc_value < 2.0:  # Low vegetable consumption
                    impact_score = feature_importance["FCVC"] * (2.0 - fcvc_value)
                    high_risk_features.append(
                        {
                            "feature": "Vegetable Intake",
                            "impact": impact_score,
                            "message": "You don't eat enough vegetables, which means missing out on important nutrients that help maintain a healthy weight",
                            "severity": "medium",
                        }
                    )

            # Sort by impact score and return top factors
            high_risk_features.sort(key=lambda x: x["impact"], reverse=True)

            # Convert to risk factor messages
            for factor in high_risk_features[:6]:  # Top 6 most impactful factors
                risk_factors.append(factor["message"])

            return (
                risk_factors
                if risk_factors
                else ["No significant risk factors identified by model analysis"]
            )

        except Exception as e:
            print(f"Error in model-based factor generation: {e}")
            return self._fallback_key_factors(data)

    def _fallback_key_factors(self, data):
        """Fallback method for generating risk factors when model analysis fails"""
        factors = []

        if data.get("smoker", False):
            factors.append(
                "Smoking can affect your metabolism and make weight management more difficult"
            )

        if data.get("family_history", False):
            factors.append(
                "Having family members with obesity increases your own risk due to shared genetics and lifestyle patterns"
            )

        if data.get("eat_between_meals") and data.get("eat_between_meals") != "no":
            factors.append(
                "Frequent snacking between meals can add extra calories to your daily intake"
            )

        if data.get("high_calorie_food", False):
            factors.append(
                "Regular consumption of high-calorie foods can lead to weight gain over time"
            )

        if data.get("physical_activity", 1.0) == 0.0:
            factors.append(
                "A sedentary lifestyle without regular exercise increases your risk of weight gain"
            )

        if data.get("technology_use", 1.0) >= 2.0:
            factors.append(
                "Spending too much time on screens often means less physical activity and more sitting"
            )

        if data.get("alcohol_consumption") in ["Frequently", "Always"]:
            factors.append(
                "Regular alcohol consumption can contribute to weight gain due to extra calories"
            )

        return factors

    def _fallback_risk_calculation(self, data):
        """Fallback method when model is not available"""
        height = data.get("height", 1.7)
        weight = data.get("weight", 70)
        bmi = self.calculate_bmi(height, weight)
        bmi_category = self.get_bmi_category(bmi)

        # Simple BMI-based risk calculation
        if bmi is None:
            base_risk = 50
        elif bmi < 18.5:
            base_risk = 15
        elif bmi < 25:
            base_risk = 25
        elif bmi < 30:
            base_risk = 45
        elif bmi < 35:
            base_risk = 65
        elif bmi < 40:
            base_risk = 80
        else:
            base_risk = 95

        return {
            "bmi": round(bmi, 1) if bmi else None,
            "bmi_category": bmi_category,
            "risk_score": base_risk,
            "risk_level": "Moderate Risk",
            "key_factors": ["Model unavailable - using simplified assessment"],
            "confidence": 70,
        }

    def generate_recommendations(self, data, risk_score, key_factors):
        """Generate intelligent recommendations based on model analysis and patient data"""
        if not self.model:
            return self._fallback_recommendations(data, risk_score)

        try:
            # Get feature importance from the model (adjusted if available)
            if hasattr(self.model, "get_adjusted_importance"):
                feature_importance = dict(
                    zip(self.feature_names, self.model.get_adjusted_importance())
                )
            else:
                feature_importance = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
            input_data = self.prepare_input_data(data)

            recommendations = []

            # Generate personalized recommendations based on model feature importance
            # and patient's current values

            # Physical Activity - Always high priority if low
            if "FAF" in feature_importance and input_data["FAF"].iloc[0] <= 1.0:
                current_level = input_data["FAF"].iloc[0]
                if current_level == 0:
                    recommendations.append(
                        "Start with light physical activity like 30-minute walks three times a week. Your body will thank you for the movement!"
                    )
                else:
                    recommendations.append(
                        "Try to get at least 150 minutes of exercise each week. Even small increases in activity can make a big difference."
                    )

            # Diet Quality - High calorie foods
            if "FAVC" in feature_importance and input_data["FAVC"].iloc[0]:
                recommendations.append(
                    "Focus on eating more whole foods like fruits, vegetables, and lean proteins instead of processed high-calorie foods."
                )

            # Family History - Genetic predisposition management
            if (
                "family_history_with_overweight" in feature_importance
                and input_data["family_history_with_overweight"].iloc[0]
            ):
                recommendations.append(
                    "Since obesity runs in your family, it's especially important to maintain healthy lifestyle habits and see your doctor regularly."
                )

            # Snacking patterns
            caec_features = [
                f
                for f in self.feature_names
                if f.startswith("CAEC_") and input_data[f].iloc[0]
            ]
            if caec_features:
                recommendations.append(
                    "Plan your meals and snacks ahead of time. Choose healthier options like fruits, nuts, or yogurt when you feel hungry between meals."
                )

            # Water consumption
            if "CH2O" in feature_importance and input_data["CH2O"].iloc[0] < 2.0:
                recommendations.append(
                    "Try to drink more water throughout the day - aim for 8-10 glasses. It helps your body function better and can reduce hunger."
                )

            # Vegetable consumption
            if "FCVC" in feature_importance and input_data["FCVC"].iloc[0] < 2.0:
                recommendations.append(
                    "Add more vegetables to your meals. Try to include them in every meal - they're packed with nutrients and help you feel full."
                )

            # Technology use / Screen time
            if "TUE" in feature_importance and input_data["TUE"].iloc[0] >= 2.0:
                recommendations.append(
                    "Take regular breaks from screens and try to move around every hour. Consider going for walks instead of watching TV."
                )

            # Smoking cessation
            if "SMOKE" in feature_importance and input_data["SMOKE"].iloc[0]:
                recommendations.append(
                    "Consider quitting smoking - it affects your metabolism and overall health. Talk to your doctor about programs that can help."
                )

            # Calorie monitoring based on current habits
            if "SCC" in feature_importance and not input_data["SCC"].iloc[0]:
                recommendations.append(
                    "Try keeping track of what you eat for a week. It can help you understand your eating patterns and make healthier choices."
                )

            # Risk-specific recommendations based on prediction
            if hasattr(self.model, "predict"):
                current_prediction = self.model.predict(input_data)[0]
            else:
                current_prediction = "Normal_Weight"  # Safe fallback

            if current_prediction in [
                "Obesity_Type_I",
                "Obesity_Type_II",
                "Obesity_Type_III",
            ]:
                recommendations.append(
                    "Consider talking to your doctor about creating a comprehensive weight management plan that's right for you."
                )
                recommendations.append(
                    "Schedule regular check-ups every 4-6 weeks to track your progress and get support along the way."
                )

            # Ensure we have at least some recommendations
            if not recommendations:
                recommendations = self._fallback_recommendations(data, risk_score)

            return recommendations

        except Exception as e:
            print(f"Error in model-based recommendation generation: {e}")
            return self._fallback_recommendations(data, risk_score)

    def _fallback_recommendations(self, data, risk_score):
        """Fallback recommendations when model analysis is unavailable"""
        recommendations = []

        if data.get("physical_activity", 1.0) <= 1.0:
            recommendations.append(
                "Try to get at least 150 minutes of moderate exercise each week - even brisk walking counts!"
            )

        if data.get("eat_between_meals") and data.get("eat_between_meals") != "no":
            recommendations.append(
                "Focus on eating balanced main meals and choose healthier snacks when you're hungry between meals"
            )

        if data.get("high_calorie_food", False):
            recommendations.append(
                "Try replacing processed high-calorie foods with whole foods like fruits, vegetables, and lean proteins"
            )

        if data.get("water_consumption", 2.0) < 2.0:
            recommendations.append(
                "Aim to drink at least 8 glasses of water daily - it helps your body function better"
            )

        if data.get("vegetable_consumption", 2.0) < 2.0:
            recommendations.append(
                "Try to include vegetables and fruits in every meal - they provide important nutrients and help you feel full"
            )

        if data.get("smoker", False):
            recommendations.append(
                "Consider quitting smoking to improve your overall health and metabolism"
            )

        if not data.get("monitor_calories", False):
            recommendations.append(
                "Try keeping a food diary for a week to understand your eating patterns better"
            )

        if data.get("technology_use", 1.0) >= 2.0:
            recommendations.append(
                "Take regular breaks from screens and try to be more active throughout the day"
            )

        if risk_score > 70:
            recommendations.append(
                "Schedule regular check-ups with your healthcare provider to monitor your progress"
            )
            recommendations.append(
                "Consider talking to a nutritionist who can help create a personalized meal plan for you"
            )

        return recommendations

    def generate_model_explanations(self, data, results):
        """Generate detailed model explanations showing why the model made its prediction"""
        # Check if model is available
        if not self.model:
            return self._fallback_model_explanations(data, results)

        # Check if feature names are available and have expected content
        try:
            if self.feature_names is None or (
                hasattr(self.feature_names, "__len__") and len(self.feature_names) < 5
            ):
                return self._fallback_model_explanations(data, results)
        except:
            return self._fallback_model_explanations(data, results)

        try:
            # Get feature importance from the model (adjusted if available)
            if hasattr(self.model, "get_adjusted_importance"):
                feature_importance = dict(
                    zip(self.feature_names, self.model.get_adjusted_importance())
                )
            elif hasattr(self.model, "feature_importances_"):
                feature_importance = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
            else:
                return self._fallback_model_explanations(data, results)
            input_data = self.prepare_input_data(data)

            if input_data is None:
                return self._fallback_model_explanations(data, results)

            # Get top important features for this model
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            top_features = sorted_features[:10]  # Top 10 most important features

            explanations = {
                "feature_impacts": [],
                "model_reasoning": "",
                "prediction_breakdown": {},
            }

            # Analyze each top feature's contribution
            bmi_processed = False  # Track if we've already processed BMI

            for feature_name, importance in top_features:
                if importance < 0.01:  # Skip very low importance features
                    continue

                # Skip Gender_Male feature if user is female (since Gender_Male=False doesn't provide meaningful insight)
                if feature_name == "Gender_Male":
                    user_gender = data.get("gender", "Unknown")
                    gender_male_value = (
                        input_data["Gender_Male"].iloc[0]
                        if "Gender_Male" in input_data.columns
                        else False
                    )

                # Handle Height/Weight as a combined BMI feature to avoid duplication
                if feature_name in ["Height", "Weight"]:
                    if bmi_processed:
                        continue  # Skip if we've already processed BMI

                    # Find the combined importance of Height and Weight
                    height_importance = feature_importance.get("Height", 0)
                    weight_importance = feature_importance.get("Weight", 0)
                    combined_importance = max(height_importance, weight_importance)

                    # Use the higher importance value for the explanation
                    feature_value = input_data[feature_name].iloc[0]
                    impact_explanation = self._explain_feature_impact(
                        feature_name, feature_value, combined_importance, data
                    )

                    if impact_explanation:
                        explanations["feature_impacts"].append(impact_explanation)

                    bmi_processed = True
                else:
                    # Handle other features normally
                    feature_value = input_data[feature_name].iloc[0]
                    impact_explanation = self._explain_feature_impact(
                        feature_name, feature_value, importance, data
                    )

                    if impact_explanation:
                        explanations["feature_impacts"].append(impact_explanation)

            # Generate overall model reasoning
            explanations["model_reasoning"] = self._generate_model_reasoning(
                results.get("prediction", ""), sorted_features, data
            )

            # Create prediction breakdown - use model's actual class order
            if hasattr(self.model, "classes_"):
                model_classes = self.model.classes_
            elif hasattr(self.model, "model") and hasattr(self.model.model, "classes_"):
                model_classes = self.model.model.classes_  # For adjusted model wrapper
            else:
                model_classes = getattr(self, "class_labels", [])

            explanations["prediction_breakdown"] = self._create_prediction_breakdown(
                results.get("probabilities", []), model_classes
            )

            return explanations

        except Exception as e:
            print(f"Error generating model explanations: {e}")
            return self._fallback_model_explanations(data, results)

    def _explain_feature_impact(self, feature_name, feature_value, importance, data):
        """Explain how a specific feature impacts the prediction"""

        # Feature name mappings for better readability
        feature_descriptions = {
            "Age": "age",
            "Height": "height",
            "Weight": "weight",
            "FAVC": "high-calorie food consumption",
            "FCVC": "vegetable consumption frequency",
            "NCP": "number of meals per day",
            "SCC": "calorie monitoring habits",
            "SMOKE": "smoking status",
            "CH2O": "daily water consumption",
            "family_history_with_overweight": "family history of obesity",
            "FAF": "physical activity frequency",
            "TUE": "technology/screen time usage",
            "Gender_Male": "gender",
            "CALC_Sometimes": "occasional alcohol consumption",
            "CALC_Frequently": "frequent alcohol consumption",
            "CAEC_Sometimes": "occasional snacking between meals",
            "CAEC_Frequently": "frequent snacking between meals",
            "CAEC_Always": "constant snacking between meals",
            "MTRANS_Automobile": "automobile transportation",
            "MTRANS_Bike": "bicycle transportation",
            "MTRANS_Motorbike": "motorbike transportation",
            "MTRANS_Walking": "walking as transportation",
        }

        readable_name = feature_descriptions.get(feature_name, feature_name)
        impact_strength = (
            "High" if importance > 0.1 else "Moderate" if importance > 0.05 else "Low"
        )

        # Generate specific explanations based on feature type and value
        if feature_name == "FAF":  # Physical Activity
            if feature_value <= 1.0:
                if impact_strength == "High":
                    explanation_text = "Your low physical activity level significantly increases obesity risk. The model weighs this heavily because regular exercise is crucial for weight management."
                elif impact_strength == "Moderate":
                    explanation_text = "Your low physical activity level contributes to increased obesity risk. The model recognizes that regular exercise is important for weight management."
                else:  # Low impact
                    explanation_text = "Your low physical activity level may contribute to obesity risk. While this factor has a smaller influence in the model, regular exercise is still beneficial for overall health."

                return {
                    "feature": "Physical Activity Level",
                    "impact": impact_strength,
                    "explanation": explanation_text,
                    "importance_score": importance,
                    "recommendation": "Increasing physical activity is one of the most effective ways to reduce your risk.",
                }
            else:
                return {
                    "feature": "Physical Activity Level",
                    "impact": "Positive",
                    "explanation": "Your regular physical activity helps reduce obesity risk. The model recognizes this as a protective factor.",
                    "importance_score": importance,
                    "recommendation": "Continue maintaining your current activity level.",
                }

        elif feature_name == "FAVC" and feature_value:  # High calorie food
            if impact_strength == "High":
                explanation_text = "The model identifies your frequent consumption of high-calorie foods as a significant risk factor. These foods are typically energy-dense and can lead to weight gain."
            elif impact_strength == "Moderate":
                explanation_text = "Your frequent consumption of high-calorie foods contributes to increased obesity risk. These foods can add extra calories to your daily intake."
            else:  # Low impact
                explanation_text = "Your consumption of high-calorie foods is noted by the model as a potential risk factor, though it has a smaller influence on the overall prediction."

            return {
                "feature": "High-Calorie Food Consumption",
                "impact": impact_strength,
                "explanation": explanation_text,
                "importance_score": importance,
                "recommendation": "Reducing high-calorie food intake could significantly lower your risk.",
            }

        elif feature_name == "family_history_with_overweight" and feature_value:
            return {
                "feature": "Family History of Obesity",
                "impact": impact_strength,
                "explanation": "Having family members with obesity increases your genetic predisposition. The model accounts for this inherited risk factor in its prediction.",
                "importance_score": importance,
                "recommendation": "While genetics can't be changed, lifestyle modifications become even more important.",
            }

        elif feature_name == "TUE" and feature_value >= 2.0:  # High screen time
            if impact_strength == "High":
                explanation_text = "Your high screen time significantly increases obesity risk through sedentary behavior, which the model weighs heavily in its assessment."
            elif impact_strength == "Moderate":
                explanation_text = "Your high screen time contributes to obesity risk through increased sedentary behavior, which the model recognizes as a concerning factor."
            else:  # Low impact
                explanation_text = "Your high screen time may contribute to obesity risk through sedentary behavior, though this has a smaller influence in the model's prediction."

            return {
                "feature": "Screen Time Usage",
                "impact": impact_strength,
                "explanation": explanation_text,
                "importance_score": importance,
                "recommendation": "Reducing screen time and taking movement breaks can help lower your risk.",
            }

        elif feature_name == "CH2O" and feature_value < 2.0:  # Low water consumption
            if impact_strength == "High":
                explanation_text = "Your low water intake significantly affects the model's risk assessment. Proper hydration is heavily weighted as important for metabolism and weight management."
            elif impact_strength == "Moderate":
                explanation_text = "Your water intake below optimal levels contributes to increased risk. The model recognizes proper hydration as important for metabolism and weight management."
            else:  # Low impact
                explanation_text = "Your water intake is below optimal levels and is noted by the model, though it has a smaller influence on the overall risk prediction."

            return {
                "feature": "Water Consumption",
                "impact": impact_strength,
                "explanation": explanation_text,
                "importance_score": importance,
                "recommendation": "Increasing water consumption may help improve your metabolic health.",
            }

        elif (
            feature_name == "FCVC" and feature_value < 2.0
        ):  # Low vegetable consumption
            if impact_strength == "High":
                explanation_text = "Your low vegetable intake significantly impacts the model's risk assessment. The model heavily weights vegetables for their nutrients and satiety benefits."
            elif impact_strength == "Moderate":
                explanation_text = "Your vegetable intake below recommended levels contributes to increased risk. The model values vegetables for their nutrients and satiety benefits."
            else:  # Low impact
                explanation_text = "Your vegetable intake is lower than recommended and is noted by the model, though it has a smaller influence on the overall prediction."

            return {
                "feature": "Vegetable Consumption",
                "impact": impact_strength,
                "explanation": explanation_text,
                "importance_score": importance,
                "recommendation": "Increasing vegetable consumption can help with weight management and overall health.",
            }

        elif feature_name.startswith("CAEC_") and feature_value:  # Eating between meals
            frequency = feature_name.split("_")[1].lower()

            if impact_strength == "High":
                explanation_text = f"Your {frequency} snacking between meals significantly increases obesity risk by adding substantial extra calories to your daily intake."
            elif impact_strength == "Moderate":
                explanation_text = f"Your {frequency} snacking between meals contributes to increased obesity risk by adding extra calories to your daily intake."
            else:  # Low impact
                explanation_text = f"Your {frequency} snacking between meals is noted by the model as adding some extra calories, though it has a smaller influence on the overall prediction."

            return {
                "feature": "Snacking Frequency",
                "impact": impact_strength,
                "explanation": explanation_text,
                "importance_score": importance,
                "recommendation": "Managing snacking habits could help reduce calorie intake and obesity risk.",
            }

        elif feature_name == "SMOKE" and feature_value:
            return {
                "feature": "Smoking Status",
                "impact": impact_strength,
                "explanation": "Smoking affects metabolism and can complicate weight management. The model includes this in its risk assessment.",
                "importance_score": importance,
                "recommendation": "Quitting smoking can improve overall health and weight management.",
            }

        elif feature_name == "Age":
            return {
                "feature": "Age Factor",
                "impact": impact_strength,
                "explanation": f"At age {int(feature_value)}, your metabolism and lifestyle patterns influence obesity risk. The model considers age-related factors in its prediction.",
                "importance_score": importance,
                "recommendation": "Age-appropriate lifestyle modifications can help maintain healthy weight.",
            }

        elif feature_name in ["Height", "Weight"]:
            bmi = data.get("weight", 70) / (data.get("height", 1.7) ** 2)
            return {
                "feature": "Body Composition (BMI)",
                "impact": impact_strength,
                "explanation": f"Your current BMI of {bmi:.1f} is a key factor in the model's assessment. BMI directly reflects the relationship between height and weight.",
                "importance_score": importance,
            }

        elif feature_name == "Gender_Male":
            # Get the actual gender from user data for contextual explanation
            user_gender = data.get("gender", "Unknown")

            return {
                "feature": f"Gender ({user_gender})",
                "impact": impact_strength,
                "explanation": f"Your gender ({user_gender}) is considered by the model as it affects metabolism, muscle mass, and fat distribution patterns. Research shows gender-specific differences in obesity risk factors.",
                "importance_score": importance,
                "recommendation": "Gender is a biological factor that can't be changed, but understanding gender-specific health patterns helps tailor recommendations.",
            }

        # Generic explanation for other features
        return {
            "feature": readable_name.title(),
            "impact": impact_strength,
            "explanation": f"This factor contributes to the model's risk assessment based on established health research.",
            "importance_score": importance,
            "recommendation": "Consult with healthcare providers about optimizing this factor.",
        }

    def _generate_model_reasoning(self, prediction, sorted_features, data):
        """Generate overall model reasoning explanation"""

        top_3_features = sorted_features[:3]
        prediction_clean = prediction.replace("_", " ") if prediction else "Unknown"

        reasoning = f"The model predicts '{prediction_clean}' based on a comprehensive analysis of your health profile. "

        if top_3_features:
            feature_names = []
            bmi_mentioned = False

            for feat_name, importance in top_3_features:
                # # Skip Gender_Male feature if user is female (similar to featur e impacts logic)
                # if feat_name == "Gender_Male":
                #     user_gender = data.get("gender", "Unknown")
                #     if user_gender == "Female":
                #         feature_names.append("gender (female)")

                if feat_name == "FAF":
                    feature_names.append("physical activity level")
                elif feat_name == "FAVC":
                    feature_names.append("dietary habits")
                elif feat_name == "family_history_with_overweight":
                    feature_names.append("family history")
                elif feat_name == "TUE":
                    feature_names.append("sedentary behavior")
                elif feat_name == "Gender_Male":
                    user_gender = data.get("gender", "Unknown")
                    feature_names.append(f"gender ({user_gender.lower()})")
                elif feat_name in ["Height", "Weight"]:
                    if not bmi_mentioned:
                        feature_names.append("BMI")
                        bmi_mentioned = True
                    # Skip adding BMI again if already mentioned
                else:
                    feature_names.append(feat_name.lower().replace("_", " "))

            if len(feature_names) >= 2:
                reasoning += f"The most influential factors in this prediction are your {', '.join(feature_names[:-1])}, and {feature_names[-1]}. "
            else:
                reasoning += f"The most influential factor is your {feature_names[0]}. "

        reasoning += "The model uses machine learning algorithms trained on extensive health datasets to identify patterns and relationships between lifestyle factors and obesity risk."

        return reasoning

    def _create_prediction_breakdown(self, probabilities, class_labels):
        """Create a breakdown of prediction probabilities"""

        # Handle numpy arrays properly
        try:
            if probabilities is None or (
                hasattr(probabilities, "__len__") and len(probabilities) == 0
            ):
                return {}
            if class_labels is None or (
                hasattr(class_labels, "__len__") and len(class_labels) == 0
            ):
                return {}
        except:
            return {}

        breakdown = {}
        for i, (prob, label) in enumerate(zip(probabilities, class_labels)):
            clean_label = label.replace("_", " ")
            breakdown[clean_label] = {
                "probability": prob * 100,
                "description": self._get_class_description(label),
            }

        return breakdown

    def _get_class_description(self, class_label):
        """Get description for each obesity class"""
        descriptions = {
            "Insufficient_Weight": "Below normal weight range, may indicate nutritional concerns",
            "Normal_Weight": "Healthy weight range with lowest obesity risk",
            "Overweight_Level_I": "Slightly above normal weight, moderate risk",
            "Overweight_Level_II": "Moderately overweight, increased health risks",
            "Obesity_Type_I": "Class I obesity, significant health risks",
            "Obesity_Type_II": "Class II obesity, high health risks",
            "Obesity_Type_III": "Class III obesity, very high health risks",
        }
        return descriptions.get(class_label, "Weight classification")

    def _fallback_model_explanations(self, data, results):
        """Fallback explanations when model analysis is not available"""
        return {
            "feature_impacts": [
                {
                    "feature": "BMI Assessment",
                    "impact": "High",
                    "explanation": "Assessment is primarily based on BMI calculation from height and weight.",
                    "importance_score": 0.8,
                    "recommendation": "Maintain healthy weight for your height.",
                }
            ],
            "model_reasoning": "This assessment uses simplified BMI-based calculations due to model unavailability. For more detailed analysis, ensure the trained model is properly loaded.",
            "prediction_breakdown": {},
        }

    def _fallback_recommendations(self, data, risk_score):
        """Fallback recommendations when model analysis is unavailable"""
        recommendations = []

        if data.get("physical_activity", 1.0) <= 1.0:
            recommendations.append(
                "Try to get at least 150 minutes of moderate exercise each week - even brisk walking counts!"
            )

        if data.get("eat_between_meals") and data.get("eat_between_meals") != "no":
            recommendations.append(
                "Focus on eating balanced main meals and choose healthier snacks when you're hungry between meals"
            )

        if data.get("high_calorie_food", False):
            recommendations.append(
                "Try replacing processed high-calorie foods with whole foods like fruits, vegetables, and lean proteins"
            )

        if data.get("water_consumption", 2.0) < 2.0:
            recommendations.append(
                "Aim to drink at least 8 glasses of water daily - it helps your body function better"
            )

        if data.get("vegetable_consumption", 2.0) < 2.0:
            recommendations.append(
                "Try to include vegetables and fruits in every meal - they provide important nutrients and help you feel full"
            )

        if data.get("smoker", False):
            recommendations.append(
                "Consider quitting smoking to improve your overall health and metabolism"
            )

        if not data.get("monitor_calories", False):
            recommendations.append(
                "Try keeping a food diary for a week to understand your eating patterns better"
            )

        if data.get("technology_use", 1.0) >= 2.0:
            recommendations.append(
                "Take regular breaks from screens and try to be more active throughout the day"
            )

        if risk_score > 70:
            recommendations.append(
                "Schedule regular check-ups with your healthcare provider to monitor your progress"
            )
            recommendations.append(
                "Consider talking to a nutritionist who can help create a personalized meal plan for you"
            )

        return recommendations


def get_discrete_risk_level(prediction):
    """Map model prediction to discrete levels for the gauge chart"""
    # Map actual model predictions to gauge positions
    prediction_mapping = {
        "Insufficient_Weight": (1, "Insufficient Weight"),
        "Normal_Weight": (2, "Normal Weight"),
        "Overweight_Level_I": (3, "Overweight Level I"),
        "Overweight_Level_II": (4, "Overweight Level II"),
        "Obesity_Type_I": (5, "Obesity Type I"),
        "Obesity_Type_II": (6, "Obesity Type II"),
        "Obesity_Type_III": (7, "Obesity Type III"),
    }

    # If it's a string prediction, use the mapping
    if isinstance(prediction, str) and prediction in prediction_mapping:
        return prediction_mapping[prediction]

    # Fallback to risk score mapping if prediction is not available
    score = prediction if isinstance(prediction, (int, float)) else 50
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


def get_bmi_gauge_position(bmi_category):
    """Map BMI category to gauge position"""
    bmi_mapping = {
        "Underweight": 1,
        "Normal weight": 2,
        "Overweight": 3,
        "Obesity Class I": 5,
        "Obesity Class II": 6,
        "Obesity Class III": 7,
        "N/A": 4,  # Default to middle if unknown
    }
    return bmi_mapping.get(bmi_category, 4)


def get_weight_class_trend(
    current_bmi_level, prediction_level, current_bmi_category, prediction_label
):
    """Determine if weight class will increase, decrease, or stay the same"""
    if prediction_level > current_bmi_level:
        return (
            f"Your weight class may increase from {current_bmi_category} to {prediction_label}",
            "#ef4444",
        )  # Red for increase
    elif prediction_level < current_bmi_level:
        return (
            f"Your weight class may decrease from {current_bmi_category} to {prediction_label}",
            "#10b981",
        )  # Green for decrease
    else:
        return (
            "Your weight class may remain unchanged in the future",
            "#3b82f6",
        )  # Blue for same


def create_risk_gauge(results):
    """Create a Plotly gauge showing both model prediction and current BMI"""

    # Get model prediction position
    if "prediction" in results and results["prediction"]:
        prediction = results["prediction"]
        prediction_level, prediction_label = get_discrete_risk_level(prediction)
    else:
        # Fallback to risk score
        risk_score = results.get("risk_score", 50)
        prediction_level, prediction_label = get_discrete_risk_level(risk_score)
        subtitle = (
            f"Risk-based assessment (Score: {risk_score})"  # Get current BMI position
        )
    bmi_category = results.get("bmi_category", "N/A")
    bmi_level = get_bmi_gauge_position(bmi_category)

    # Get weight class trend message and color
    trend_message, trend_color = get_weight_class_trend(
        bmi_level, prediction_level, bmi_category, prediction_label
    )

    # Create the base gauge with prediction needle
    fig = go.Figure()

    # Add the main gauge indicator (for model prediction)
    fig.add_trace(
        go.Indicator(
            mode="gauge",
            value=prediction_level,
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
                    "tickfont": {"color": "white", "size": 10},
                },
                "bar": {"color": "rgba(0,0,0,0)"},  # Hide the default bar
                "steps": [
                    {
                        "range": [0.5, 1.5],
                        "color": "#38bdf8",
                    },  # Level 1: Insufficient Weight (light blue - underweight)
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
                # Add custom needle markers using shapes
            },
        )
    )

    # Add custom needle shapes for both indicators
    # Prediction needle (blue)
    prediction_angle = 180 - (prediction_level - 0.5) * (180 / 7)
    prediction_x = 0.5 + 0.35 * np.cos(np.radians(prediction_angle))
    prediction_y = 0.25 + 0.35 * np.sin(np.radians(prediction_angle))

    # BMI needle (red)
    bmi_angle = 180 - (bmi_level - 0.5) * (180 / 7)
    bmi_x = 0.5 + 0.30 * np.cos(np.radians(bmi_angle))
    bmi_y = 0.25 + 0.30 * np.sin(np.radians(bmi_angle))

    # Determine the color for the BMI needle and legend dynamically
    bmi_colors = [
        "#38bdf8",  # Insufficient Weight (light blue)
        "#10b981",  # Normal Weight (green)
        "#fbbf24",  # Overweight Level I (yellow)
        "#f59e0b",  # Overweight Level II (amber)
        "#f97316",  # Obesity Type I (orange)
        "#ef4444",  # Obesity Type II (red)
        "#b91c1c",  # Obesity Type III (dark red)
    ]
    bmi_needle_color = bmi_colors[int(bmi_level) - 1]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        # Add needle shapes
        shapes=[
            # Prediction needle (blue - longer)
            dict(
                type="line",
                x0=0.5,
                y0=0.25,
                x1=prediction_x,
                y1=prediction_y,
                xref="paper",
                yref="paper",
                line=dict(color="#3b82f6", width=10),
            ),
            # BMI needle (dynamic color)
            dict(
                type="line",
                x0=0.5,
                y0=0.25,
                x1=bmi_x,
                y1=bmi_y,
                xref="paper",
                yref="paper",
                line=dict(
                    color=bmi_needle_color,
                    width=10,
                ),
            ),
            # Center circle
            dict(
                type="circle",
                x0=0.48,
                y0=0.22,
                x1=0.52,
                y1=0.31,
                xref="paper",
                yref="paper",
                fillcolor="white",
                line=dict(color="white", width=6),
            ),
        ],
        # Text annotations
        annotations=[
            dict(
                text=f"<b>{trend_message}</b>",
                x=0.5,
                y=-0.02,
                xref="paper",
                yref="paper",
                font=dict(size=12, color=trend_color),
                showarrow=False,
                align="center",
            ),
            dict(
                text=f"{prediction_label}",
                x=0.5,
                y=0.03,
                xref="paper",
                yref="paper",
                font=dict(size=20, color=bmi_colors[int(prediction_level) - 1]),
                showarrow=False,
                align="center",
            ),
            dict(
                text=f"<b>Future Weight Prediction:</b>",
                x=0.5,
                y=0.1,
                xref="paper",
                yref="paper",
                font=dict(size=20, color="white"),
                showarrow=False,
                align="center",
            ),
            # Legend for needles
            dict(
                text="Model Future Prediction <span style='color:#3b82f6'>●</span> ",
                x=1,
                y=1,
                xref="paper",
                yref="paper",
                font=dict(size=12, color="white"),
                showarrow=False,
                align="left",
            ),
            dict(
                text=f"Current BMI <span style='color:{bmi_needle_color}'>●</span>",
                x=1,
                y=0.95,
                xref="paper",
                yref="paper",
                font=dict(size=12, color="white"),
                showarrow=False,
                align="left",
            ),
        ],
    )

    return fig


def render_assessment_page():
    assessment = ObesityAssessment()

    # Header
    st.markdown(
        """
            # :material/routine: HabitIQ - Obesity Risk Assessment
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="nav-button-left">', unsafe_allow_html=True)
        if st.session_state.current_step > 1:
            if st.button("← Previous", key="prev_btn"):
                st.session_state.current_step -= 1
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="nav-button-right">', unsafe_allow_html=True)
        if st.session_state.current_step < assessment.total_steps:
            if st.button("Next →", key="next_btn"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button(
                ":material/rocket_launch: Generate Assessment", key="generate_btn"
            ):
                generate_assessment()
        st.markdown("</div>", unsafe_allow_html=True)


def render_step_1():
    st.markdown(
        """
            ### :material/account_circle: Personal Information
            Please provide the patient's personal information. The data will be used to assess the patient's obesity risk and generate personalized recommendations.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=28,
            key="age_input",
            help="Enter the patient's current age in years. This helps assess age-related obesity risk factors.",
        )
    with col2:
        height = st.number_input(
            "Height (cm)",
            min_value=100,
            max_value=250,
            value=175,
            key="height_input",
            help="Enter the patient's height in centimeters. This is used to calculate their BMI and assess weight status.",
        )

    with col3:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0,
            key="gender_input",
            help="Select the patient's biological gender. This affects metabolic rate and obesity risk assessment.",
        )

    with col4:
        weight = st.number_input(
            "Weight (kg)",
            min_value=30,
            max_value=300,
            value=76,
            key="weight_input",
            help="Enter the patient's current weight in kilograms. Combined with height, this calculates their BMI. ",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "age": float(age),
            "gender": gender,
            "height": float(height / 100),
            "weight": float(weight),
        }
    )


def render_step_2():

    st.markdown(
        """
            ### :material/directions_bike: Lifestyle & Habits
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        physical_activity = st.selectbox(
            "Physical Activity Frequency",
            ["Never", "1-2 days per week", "2-4 days per week", "4-5 days per week"],
            index=0,  # Default to "Never"
            key="physical_activity_input",
            help="Select how often the patient engages in physical exercise or sports activities per week.",
        )
    with col2:
        transportation = st.selectbox(
            "Primary Transportation Mode",
            ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"],
            index=0,  # Default to Automobile
            key="transportation_input",
            help="Choose the patient's main method of transportation for daily activities like work or shopping.",
        )

    with col3:
        technology_use = st.selectbox(
            "Technology Use (screen time hours/day)",
            ["0-2 hours", "3-5 hours", "More than 5 hours"],
            index=2,  # Default to "More than 5 hours"
            key="technology_use_input",
            help="Select the average number of hours the patient spends daily on screens (TV, computer, phone, tablets, video games).",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "physical_activity": {
                "Never": 0.0,
                "1-2 days per week": 1.0,
                "2-4 days per week": 2.0,
                "4-5 days per week": 3.0,
            }[physical_activity],
            "transportation": (
                "Public_Transportation"
                if transportation == "Public Transportation"
                else transportation
            ),
            "technology_use": {
                "0-2 hours": 0.0,
                "3-5 hours": 1.0,
                "More than 5 hours": 2.0,
            }[technology_use],
        }
    )


def render_step_3():
    st.markdown(
        """  
            ### :material/restaurant_menu: Dietary Habits
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col2:
        meals_per_day = st.selectbox(
            "Number of Main Meals per Day",
            ["Between 1 & 2", "Exactly 3", "More than 3"],
            index=0,  # Default to "Between 1 & 2"
            key="meals_per_day_input",
            help="Select how many full meals the patient typically eats per day (breakfast, lunch, dinner, etc.).",
        )
    with col3:
        water_consumption = st.selectbox(
            "Water Consumption (litres/day)",
            ["Less than a litre", "Between 1 and 2L", "More than 2L"],
            index=0,  # Default to "Less than a litre"
            key="water_consumption_input",
            help="Select the number of litres of water the patient drinks daily. Include water from all sources.",
        )

    with col4:
        vegetable_consumption = st.selectbox(
            "Vegetable Consumption",
            ["Never", "Sometimes", "Always"],
            index=0,  # Default to "Never"
            key="vegetable_consumption_input",
            help="Select how often the patient eats vegetables in their meals.",
        )
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            ["Never", "Sometimes", "Frequently", "Always"],
            index=2,  # Default to "Frequently"
            key="alcohol_consumption_input",
            help="Select how often the patient consumes alcoholic beverages. Consider beer, wine, spirits, and mixed drinks.",
        )
    with col3:
        eat_between_meals = st.selectbox(
            "Do you tend to eat between meals?",
            key="eat_between_meals_input",
            options=["No", "Sometimes", "Frequently", "Always"],
            index=3,  # Default to "Always"
            help="Check if the patient regularly snacks or eats food between their main meals throughout the day.",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "meals_per_day": {
                "Between 1 & 2": 1.0,
                "Exactly 3": 2.0,
                "More than 3": 3.0,
            }[meals_per_day],
            "water_consumption": {
                "Less than a litre": 1.0,
                "Between 1 and 2L": 2.0,
                "More than 2L": 3.0,
            }[water_consumption],
            "vegetable_consumption": {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}[
                vegetable_consumption
            ],
            "eat_between_meals": (
                "no" if eat_between_meals == "No" else eat_between_meals
            ),
            "alcohol_consumption": (
                "no" if alcohol_consumption == "Never" else alcohol_consumption
            ),
        }
    )


def render_step_4():
    st.markdown(
        """            
            ### :material/health_and_safety: Health Factors
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        smoker = st.checkbox(
            "Patient is a smoker.",
            value=True,  # Default to checked
            key="smoker_input",
            help="Check if the patient currently smokes cigarettes, cigars, or uses other tobacco products regularly.",
        )

        family_history = st.checkbox(
            "Patient has family member(s) who suffer from obesity.",
            value=True,  # Default to checked
            key="family_history_input",
            help="Check if any of the patient's immediate family members (parents, siblings) have a history of obesity or weight problems.",
        )

    with col2:
        high_calorie_food = st.checkbox(
            "Patient frequently tend to consume high-calorie foods.",
            value=True,  # Default to checked
            key="high_calorie_food_input",
            help="Check if the patient regularly eats foods high in calories like fast food, sweets, fried foods, or processed snacks.",
        )

        monitor_calories = st.checkbox(
            "Patient monitors their calorie intake daily.",
            value=False,  # Default to unchecked
            key="monitor_calories_input",
            help="Check if the patient actively tracks or monitors the number of calories they consume daily through apps or food logs.",
        )

    # Save to session state for step transitions
    st.session_state.patient_data.update(
        {
            "smoker": smoker,
            "family_history": family_history,
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
    model_explanations = assessment.generate_model_explanations(
        st.session_state.patient_data, results
    )
    results["recommendations"] = recommendations
    results["model_explanations"] = model_explanations
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

    # Show model status with debugging info
    assessment = ObesityAssessment()
    if assessment.model is None:
        st.error("❌ **Model not available** - using fallback assessment")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ## :material/balance: Future Obesity Risk VS Current BMI
        """
        )

        # Plotly gauge chart (outside HTML but visually inside)
        gauge_fig = create_risk_gauge(results)
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
                age=int(data["age"]),
                gender=data["gender"],
                height=int(data["height"] * 100),
                weight=int(data["weight"]),
            ),
            unsafe_allow_html=True,
        )

        # Model Confidence Card
        # Convert confidence percentage (0-100) to 1-10 scale
        confidence_score = round(results["confidence"] / 10)
        st.markdown(
            f"""
        <div class="custom-card">
            <div class="custom-card-title" style="align-items: center; justify-content: center; display: flex;">
                    <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle;">bolt</span>
                    <h3 style="font-weight: 600; margin: 0; text-align: center;">Model Confidence</h3>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">{confidence_score}/10</div>
                <p style="color: #9ca3af; font-size: 0.8rem;">(How sure the model is of its prediction - higher scores indicate greater certainty)</p>
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
                ":material/description: Generate Report",
                key="report_btn",
                use_container_width=True,
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
            <div class="risk-factor-item">
                <span class="material-symbols-outlined" style="font-size: 2.2rem; vertical-align: middle; color: #ef4444;">warning</span> 
                <p style="font-size: 1rem; margin: 0; margin-top: 0.25rem; color: #ffffff;">{factor}</p>
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

    # Model Explanations Section
    st.markdown(
        """
            ## :material/neurology: Model Explanations
        """
    )

    explanations = results.get("model_explanations", {})

    # Overall model reasoning
    if explanations.get("model_reasoning"):
        st.markdown(
            f"""
        <div class="custom-card">
            <div class="custom-card-title">
                <span class="material-symbols-outlined" style="font-size: 1.5rem;">psychology</span>
                <h4 style="margin: 0;">How the Model Made This Prediction</h4>
            </div>
            <p style="font-size: 1rem; line-height: 1.6; margin: 0;">{explanations["model_reasoning"]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature Impact Analysis
    feature_impacts = explanations.get("feature_impacts", [])
    if feature_impacts:
        st.markdown(
            "### :material/track_changes: **Key Features Influencing Your Risk**"
        )

        # Sort by importance and show top 6 factors
        sorted_impacts = sorted(
            feature_impacts, key=lambda x: x.get("importance_score", 0), reverse=True
        )

        # Create bar chart for feature importance
        if len(sorted_impacts) > 1:
            import pandas as pd
            import altair as alt

            # Prepare data for Altair chart with proper sorting
            chart_impacts = sorted_impacts[:8]  # Top 8 most important features

            # Create DataFrame for Altair
            chart_data = []
            for impact in chart_impacts:
                chart_data.append(
                    {
                        "Feature": impact["feature"],
                        "Importance Score": impact.get("importance_score", 0) * 100,
                        "Impact Level": impact["impact"],
                    }
                )

            df = pd.DataFrame(chart_data)

            # Create Altair horizontal bar chart with proper sorting
            chart = (
                alt.Chart(df)
                .mark_bar(color="#3b82f6")
                .add_selection(alt.selection_interval())
                .encode(
                    x=alt.X("Importance Score:Q", title="Importance Score (%)"),
                    y=alt.Y(
                        "Feature:N",
                        sort=alt.EncodingSortField(
                            field="Importance Score", order="descending"
                        ),
                        title="Feature",
                    ),
                    tooltip=["Feature:N", "Importance Score:Q", "Impact Level:N"],
                )
                .properties(
                    height=max(300, len(chart_impacts) * 40),
                    title="Feature Importance Ranking",
                )
            )

            # Display the chart
            st.altair_chart(chart, use_container_width=True)

        for i, impact in enumerate(sorted_impacts[:6]):
            impact_color = (
                "#ef4444"
                if impact["impact"] == "High"
                else "#f59e0b" if impact["impact"] == "Moderate" else "#10b981"
            )

            st.markdown(
                f"""
            <div class="custom-card" style="border-left: 4px solid {impact_color};">
                <div style="display: flex; justify-content: between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0; color: {impact_color};">{impact["feature"]}</h4>
                        <span style="background: {impact_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 1rem; font-size: 0.75rem; font-weight: bold;">
                            {impact["impact"]} Impact
                        </span>
                    </div>
                    <div style="text-align: right; color: #9ca3af; font-size: 0.875rem;">
                        Importance: {impact.get("importance_score", 0):.1%}
                    </div>
                </div>
                <p style="margin: 0.5rem 0; line-height: 1.5;">{impact["explanation"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Prediction Breakdown
    prediction_breakdown = explanations.get("prediction_breakdown", {})
    if prediction_breakdown:
        st.markdown("### :material/pie_chart: **Prediction Probability Breakdown**")

        # Prepare data for pie chart
        labels = []
        values = []
        colors = []

        # Sort by probability for better visualization
        sorted_predictions = sorted(
            prediction_breakdown.items(),
            key=lambda x: x[1]["probability"],
            reverse=True,
        )

        for class_name, data in sorted_predictions:
            probability = data["probability"]
            if (
                probability > 0.1
            ):  # Only show classes with >0.1% probability for cleaner chart
                labels.append(class_name)
                values.append(probability)

                # Assign colors based on risk level
                if "Normal Weight" in class_name:
                    colors.append("#10b981")  # Green
                elif "Insufficient" in class_name:
                    colors.append("#fbbf24")  # Yellow
                elif "Overweight Level I" in class_name:
                    colors.append("#f59e0b")  # Amber
                elif "Overweight Level II" in class_name:
                    colors.append("#f97316")  # Orange
                elif "Obesity Type I" in class_name:
                    colors.append("#ef4444")  # Red
                elif "Obesity Type II" in class_name:
                    colors.append("#dc2626")  # Dark Red
                elif "Obesity Type III" in class_name:
                    colors.append("#b91c1c")  # Very Dark Red
                else:
                    colors.append("#6b7280")  # Gray

        if labels and values:
            # Create pie chart
            fig = go.Figure()

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors, line=dict(color="#1f2937", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>%{label}</b><br>Probability: %{value:.1f}%<br>Description: %{customdata}<extra></extra>",
                    customdata=[
                        prediction_breakdown[label]["description"] for label in labels
                    ],
                    pull=[
                        0.1 if i == 0 else 0 for i in range(len(labels))
                    ],  # Pull out the highest probability slice
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=500,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(color="white"),
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

    # Fallback if no explanations available
    if not any(
        [
            explanations.get("model_reasoning"),
            feature_impacts,
            prediction_breakdown,
        ]
    ):
        st.info(
            """
        **Model explanations are not available.** This could be because:
        - The trained model file is not accessible
        - Model analysis encountered an error
        - Using simplified assessment mode
        
        For detailed model insights, ensure the Random Forest model is properly loaded.
        """
        )


def convert_to_display_values(data):
    """Convert encoded values back to human-readable format for display in reports"""
    display_data = data.copy()

    # Physical Activity mapping (reverse)
    physical_activity_map = {
        0.0: "Never",
        1.0: "1-2 days per week",
        2.0: "2-4 days per week",
        3.0: "4-5 days per week",
    }
    if "physical_activity" in display_data:
        display_data["physical_activity"] = physical_activity_map.get(
            display_data["physical_activity"], "Not specified"
        )

    # Technology Use mapping (reverse)
    technology_use_map = {0.0: "0-2 hours", 1.0: "3-5 hours", 2.0: "More than 5 hours"}
    if "technology_use" in display_data:
        display_data["technology_use"] = technology_use_map.get(
            display_data["technology_use"], "Not specified"
        )

    # Transportation mapping (reverse)
    if "transportation" in display_data:
        if display_data["transportation"] == "Public_Transportation":
            display_data["transportation"] = "Public Transportation"
        # Other transportation values remain the same

    # Meals per day mapping (reverse)
    meals_per_day_map = {1.0: "Between 1 & 2", 2.0: "Exactly 3", 3.0: "More than 3"}
    if "meals_per_day" in display_data:
        display_data["meals_per_day"] = meals_per_day_map.get(
            display_data["meals_per_day"], "Not specified"
        )

    # Water consumption mapping (reverse)
    water_consumption_map = {
        1.0: "Less than a litre",
        2.0: "Between 1 and 2L",
        3.0: "More than 2L",
    }
    if "water_consumption" in display_data:
        display_data["water_consumption"] = water_consumption_map.get(
            display_data["water_consumption"], "Not specified"
        )

    # Vegetable consumption mapping (reverse)
    vegetable_consumption_map = {1.0: "Never", 2.0: "Sometimes", 3.0: "Always"}
    if "vegetable_consumption" in display_data:
        display_data["vegetable_consumption"] = vegetable_consumption_map.get(
            display_data["vegetable_consumption"], "Not specified"
        )

    # Alcohol consumption mapping (reverse)
    if "alcohol_consumption" in display_data:
        if display_data["alcohol_consumption"] == "no":
            display_data["alcohol_consumption"] = "Never"
        # Other values (Sometimes, Frequently, Always) remain the same

    # Eat between meals mapping (reverse)
    if "eat_between_meals" in display_data:
        if display_data["eat_between_meals"] == "no":
            display_data["eat_between_meals"] = "No"
        # Other values (Sometimes, Frequently, Always) remain the same

    # Height and weight formatting
    if "height" in display_data:
        display_data["height"] = f"{int(display_data['height'] * 100)} cm"
    if "weight" in display_data:
        display_data["weight"] = f"{int(display_data['weight'])} kg"
    if "age" in display_data:
        display_data["age"] = f"{int(display_data['age'])} years"

    return display_data


def render_report_page():
    results = st.session_state.assessment_results
    data = st.session_state.patient_data
    report_date = datetime.now().strftime("%B %d, %Y")

    # Convert data to display format
    display_data = convert_to_display_values(data)

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
            value=display_data["age"],
            label="Age",
            delta=None,
            help="Patient's age in years",
        )
    with col2:
        st.metric(
            value=display_data["gender"],
            label="Gender",
            delta=None,
            help="Patient's gender",
        )
    with col3:
        st.metric(
            value=display_data["height"],
            label="Height",
            delta=None,
            help="Patient's height in cm",
        )
    with col4:
        st.metric(
            value=display_data["weight"],
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
            help="Overall risk level based on comprehensive assessment",
        )

    # Use model prediction if available, otherwise fall back to risk score
    if "prediction" in results and results["prediction"]:
        _, level_label = get_discrete_risk_level(results["prediction"])
        prediction_source = "AI Model Prediction"
        prediction_help = "Future weight classification predicted by the trained AI model based on lifestyle patterns"
    else:
        _, level_label = get_discrete_risk_level(results["risk_score"])
        prediction_source = "Risk-Based Assessment"
        prediction_help = "Weight classification based on calculated risk score from lifestyle factors"

    with col2:
        st.metric(
            value=level_label,
            label=prediction_source,
            delta=None,
            help=prediction_help,
        )

    with col3:
        # Add model confidence if available
        if "confidence" in results and results["confidence"]:
            confidence_score = round(results["confidence"] / 10)
            st.metric(
                value=f"{confidence_score}/10",
                label="Model Confidence",
                delta=None,
                help="How confident the AI model is in its prediction (higher = more certain)",
            )
        else:
            st.metric(
                value=results["bmi_category"],
                label="Current BMI Category",
                delta=None,
                help="Current BMI category based on height and weight",
            )

    st.markdown("---")

    # Model Predictions vs Current Status (if model prediction available)
    if "prediction" in results and results["prediction"]:
        st.markdown(
            """
            ## :material/psychology: Model Predictions vs Current Status
        """,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### **Current Status**")
            st.metric(
                label="Current BMI Category",
                value=results["bmi_category"],
                help="Your current BMI classification based on height and weight",
            )
            if results.get("bmi"):
                st.metric(
                    label="Current BMI Value",
                    value=f"{results['bmi']:.1f}",
                    help="Your calculated Body Mass Index",
                )

        with col2:
            st.markdown("### **AI Model Prediction**")
            prediction_clean = results["prediction"].replace("_", " ")
            st.metric(
                label="Predicted Future Classification",
                value=prediction_clean,
                help="AI model's prediction of your future weight classification based on current lifestyle patterns",
            )
            if results.get("confidence"):
                st.metric(
                    label="Prediction Confidence",
                    value=f"{results['confidence']:.1f}%",
                    help="How confident the AI model is in this prediction",
                )

        # Add interpretation
        current_bmi_level = get_bmi_gauge_position(results["bmi_category"])
        prediction_level, _ = get_discrete_risk_level(results["prediction"])

        if prediction_level > current_bmi_level:
            trend_message = "⚠️ **The model predicts your weight classification may increase** based on current lifestyle patterns. Consider implementing the recommendations below."
            st.warning(trend_message)
        elif prediction_level < current_bmi_level:
            trend_message = "✅ **The model predicts your weight classification may improve** if you maintain current positive habits."
            st.success(trend_message)
        else:
            trend_message = "ℹ️ **The model predicts your weight classification will likely remain stable** based on current patterns."
            st.info(trend_message)

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
            value=display_data.get("physical_activity", "Not specified"),
            help="How often the patient engages in physical activity",
        )
        st.metric(
            label="Primary Transportation",
            value=display_data.get("transportation", "Not specified"),
            help="Main mode of transportation",
        )
        st.metric(
            label="Screen Time (hrs/day)",
            value=display_data.get("technology_use", "Not specified"),
            help="Average daily technology/screen usage",
        )

    with col2:
        st.markdown("### **Dietary Habits**")
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="Meals/day",
                value=display_data.get("meals_per_day", "Not specified"),
            )
            st.metric(
                label="Water intake",
                value=display_data.get("water_consumption", "Not specified"),
                help="Average daily water consumption",
            )
        with col4:
            st.metric(
                label="Vegetables",
                value=display_data.get("vegetable_consumption", "Not specified"),
                help="Average daily vegetable servings",
            )
            st.metric(
                label="Alcohol",
                value=display_data.get("alcohol_consumption", "Not specified"),
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
