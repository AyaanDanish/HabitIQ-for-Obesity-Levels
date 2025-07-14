#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
#%%

df = pd.read_csv("../data/ObesityData.csv")
df.head()

#%%
# Reduce BMI influence in feature importance analysis:
def adjust_feature_importance_for_prediction(model, feature_names):
    importance = model.feature_importances_
    
    # Identify BMI-related features
    bmi_features = ['Height', 'Weight']
    
    # Redistribute BMI importance to lifestyle factors
    bmi_indices = [i for i, name in enumerate(feature_names) if name in bmi_features]
    lifestyle_indices = [i for i, name in enumerate(feature_names) 
                        if name in ['FAF', 'FAVC', 'TUE', 'family_history_with_overweight']]
    
    # Calculate adjustment factor
    bmi_total_importance = sum(importance[i] for i in bmi_indices)
    adjustment_factor = 0.6  # Reduce BMI importance by 40%
    
    # Adjust importance for explanations
    adjusted_importance = importance.copy()
    
    # Reduce BMI importance
    for i in bmi_indices:
        adjusted_importance[i] *= adjustment_factor
    
    # Boost lifestyle factor importance
    boost_amount = bmi_total_importance * (1 - adjustment_factor) / len(lifestyle_indices)
    for i in lifestyle_indices:
        adjusted_importance[i] += boost_amount
    
    return adjusted_importance

#%%
binary_cols = ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight',]

# Map binary columns to booleans
for col in binary_cols:
    unique_vals = sorted(df[col].unique())
    # Map the first unique value to False, the second to True
    mapping = {unique_vals[0]: False, unique_vals[1]: True}
    df[col] = df[col].map(mapping)

df[binary_cols].head()

#%%
# One-hot encode all object columns
X = pd.get_dummies(df.drop('NObeyesdad', axis=1), drop_first=True)
y = df['NObeyesdad']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_features=None)
model.fit(X_train, y_train)

#%%
# Call the adjustment function to get rebalanced feature importance
feature_names = X.columns.tolist()
adjusted_importance = adjust_feature_importance_for_prediction(model, feature_names)

# Compare original vs adjusted importance
print("=== FEATURE IMPORTANCE COMPARISON ===")
print("Original vs Adjusted Importance for Key Features:\n")

importance_comparison = pd.DataFrame({
    'Feature': feature_names,
    'Original_Importance': model.feature_importances_,
    'Adjusted_Importance': adjusted_importance,
    'Change': adjusted_importance - model.feature_importances_
})

# Sort by adjusted importance
importance_comparison = importance_comparison.sort_values('Adjusted_Importance', ascending=False)

# Show top 10 most important features
print("TOP 10 FEATURES (Adjusted Importance):")
print(importance_comparison.head(10).to_string(index=False, float_format='%.4f'))

print("\n=== KEY CHANGES ===")
# Show BMI-related changes
bmi_features = importance_comparison[importance_comparison['Feature'].isin(['Height', 'Weight'])]
print("\nBMI-Related Features (Reduced):")
print(bmi_features[['Feature', 'Original_Importance', 'Adjusted_Importance', 'Change']].to_string(index=False, float_format='%.4f'))

# Show lifestyle feature changes
lifestyle_features = importance_comparison[importance_comparison['Feature'].isin(['FAF', 'FAVC', 'TUE', 'family_history_with_overweight'])]
print("\nLifestyle Features (Boosted):")
print(lifestyle_features[['Feature', 'Original_Importance', 'Adjusted_Importance', 'Change']].to_string(index=False, float_format='%.4f'))

#%%
# Create visualization of the adjustment
plt.figure(figsize=(15, 8))

# Get top 12 features for visualization
top_features = importance_comparison.head(12)

x = np.arange(len(top_features))
width = 0.35

plt.bar(x - width/2, top_features['Original_Importance'], width, 
        label='Original Importance', alpha=0.8, color='lightcoral')
plt.bar(x + width/2, top_features['Adjusted_Importance'], width, 
        label='Adjusted Importance', alpha=0.8, color='lightblue')

plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance: Original vs Adjusted\n(Reducing BMI Dominance, Boosting Lifestyle Factors)')
plt.xticks(x, top_features['Feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Save the adjusted importance for use in the app
import pickle

# Create a wrapper class to store both model and adjusted importance
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

# Create the adjusted model wrapper
adjusted_model = AdjustedImportanceModel(model, feature_names, adjusted_importance)

# Save the adjusted model
with open('../model/obesity_model_adjusted.pkl', 'wb') as f:
    pickle.dump(adjusted_model, f)

print("\n=== MODEL SAVED ===")
print("Adjusted model saved to '../model/obesity_model_adjusted.pkl'")
print("This model will show lifestyle factors as more important for future risk prediction.")

# Evaluate
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
y_pred = model.predict(X_test)
print("\n=== MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred))
#%%