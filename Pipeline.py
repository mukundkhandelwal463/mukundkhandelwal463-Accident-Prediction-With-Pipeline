import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load and preprocess data
data = pd.read_csv(
    "C:\\Users\\mukun\\OneDrive\\Documents\\python\\sklearn\\sklearn_tutorials\\class 6\\10 pipe dataset.csv")
data["Time"] = pd.to_datetime(data["Time"], format="%H:%M:%S", errors="coerce")
data["Hour_of_Day"] = data["Time"].dt.hour
new_df = data.drop("Time", axis=1)

# Encode target
lb = LabelEncoder()
new_df['Accident_severity'] = lb.fit_transform(new_df['Accident_severity'])

# Balance data
X = new_df.drop(columns=['Accident_severity'])
y = new_df['Accident_severity']
X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation strategies
strategies = {
    3: 'most_frequent', 4: 'most_frequent', 5: 'most_frequent', 6: 'most_frequent',
    8: 'constant', 9: 'constant', 10: 'most_frequent', 11: 'most_frequent', 12: 'most_frequent',
    13: 'most_frequent', 14: 'most_frequent', 18: 'most_frequent', 21: 'most_frequent',
    26: 'most_frequent', 27: 'most_frequent'
}

# Transformers
tf1 = ColumnTransformer([
    (f'impute_{i}', SimpleImputer(strategy=strategies[i], fill_value='Unknown'), [i]) for i in strategies
], remainder='passthrough')

object_columns_indices = list(range(31))
tf2 = ColumnTransformer([
    (f'ohe_{col}', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [col])
    for col in object_columns_indices
], remainder='passthrough')

tf4 = SelectKBest(chi2, k=50)
model = RandomForestClassifier()

pipe = Pipeline([
    ('trf1', tf1),
    ('trf2', tf2),
    ('trf4', tf4),
    ('model', model)
])
pipe.fit(X_train, y_train)


# Prediction function
def predict_accident(features):
    return pipe.predict(np.array([features]))[0], pipe.predict_proba(np.array([features]))


# Streamlit App
st.markdown("<h1 style='text-align: center; color: black;'>Accident Prediction With Pipeline</h1>", unsafe_allow_html=True)
st.image("10 pipe img.png", use_container_width=True)

st.sidebar.header("Enter Accident Details")

# Sidebar Inputs
input_data = {
    "Day_of_week": st.sidebar.selectbox("Day of Week",
                                        ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']),
    "Age_band_of_driver": st.sidebar.selectbox("Driver Age Band", ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown']),
    "Sex_of_driver": st.sidebar.selectbox("Sex of Driver", ['Male', 'Female', 'Unknown']),
    "Educational_level": st.sidebar.selectbox("Educational Level",
                                              ['None', 'Primary', 'Junior high school', 'High school',
                                               'Above high school']),
    "Vehicle_driver_relation": st.sidebar.selectbox("Driver Relation to Vehicle", ['Owner', 'Employee', 'Unknown']),
    "Driving_experience": st.sidebar.selectbox("Driving Experience",
                                               ['No Licence', '1-2yr', '2-5yr', 'Above 10yr', 'Unknown']),
    "Type_of_vehicle": st.sidebar.selectbox("Vehicle Type",
                                            ['Automobile', 'Lorry', 'Taxi', 'Public bus', 'Long lorry']),
    "Owner_of_vehicle": st.sidebar.selectbox("Owner of Vehicle", ['Owner', 'Organization']),
    "Service_year_of_vehicle": st.sidebar.selectbox("Service Year",
                                                    ['0-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Unknown']),
    "Defect_of_vehicle": st.sidebar.selectbox("Vehicle Defect", ['No defect', 'Brakes', 'Unknown']),
    "Area_accident_occured": st.sidebar.selectbox("Accident Area", ['Residential areas', 'Other', 'Office areas']),
    "Lanes_or_Medians": st.sidebar.selectbox("Lanes",
                                             ['Undivided Two way', 'Two-way (divided with solid lines road marking)']),
    "Road_allignment": st.sidebar.selectbox("Road Alignment", ['Tangent road with flat terrain', 'Unknown']),
    "Types_of_Junction": st.sidebar.selectbox("Junction Type", ['No junction', 'Y Shape', 'Unknown']),
    "Road_surface_type": st.sidebar.selectbox("Surface Type", ['Asphalt', 'Earth', 'Gravel', 'Unknown']),
    "Road_surface_conditions": st.sidebar.selectbox("Surface Condition", ['Dry', 'Wet or damp']),
    "Light_conditions": st.sidebar.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit']),
    "Weather_conditions": st.sidebar.selectbox("Weather", ['Normal', 'Rainy', 'Cloudy']),
    "Type_of_collision": st.sidebar.selectbox("Collision Type",
                                              ['Vehicle with vehicle', 'Collision with animals', 'Unknown']),
    "Number_of_vehicles_involved": st.sidebar.slider("Vehicles Involved", 1, 5, 2),
    "Number_of_casualties": st.sidebar.slider("Casualties", 1, 10, 1),
    "Vehicle_movement": st.sidebar.selectbox("Movement", ['Going straight', 'Stopping', 'Turning right']),
    "Casualty_class": st.sidebar.selectbox("Casualty Class", ['Driver or rider', 'Passenger']),
    "Sex_of_casualty": st.sidebar.selectbox("Casualty Sex", ['Male', 'Female']),
    "Age_band_of_casualty": st.sidebar.selectbox("Casualty Age Band", ['Under 18', '18-30', '31-50', 'Over 51']),
    "Casualty_severity": st.sidebar.slider("Casualty Severity", 1, 3, 2),
    "Work_of_casuality": st.sidebar.selectbox("Occupation", ['Driver', 'Passenger']),
    "Fitness_of_casuality": st.sidebar.selectbox("Fitness", ['Normal', 'Deaf', 'Other']),
    "Pedestrian_movement": st.sidebar.selectbox("Pedestrian Movement", ['Not a Pedestrian', 'Crossing']),
    "Cause_of_accident": st.sidebar.selectbox("Cause", ['Changing lane to the left', 'Overspeed', 'Unknown']),
    "Hour_of_Day": st.sidebar.slider("Hour of Day", 0, 23, 12),
}

if st.sidebar.button("Predict"):
    features = list(input_data.values())
    result, proba = predict_accident(features)
    severity_map = {0: "Fatal Injury", 1: "Serious Injury", 2: "Slight Injury"}
    st.success(f"🚑 Predicted Severity: **{severity_map[result]}**")

    # Add confidence chart
    st.markdown("### 🔍 Prediction Confidence")
    proba_df = pd.DataFrame(proba, columns=[severity_map[i] for i in range(3)])
    st.bar_chart(proba_df.T)

    # Optional: Feature importance chart
    st.markdown("### 📊 Feature Importance")
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(10, 4))
        sns.barplot(x=importances[indices], y=np.array(X.columns)[indices])
        plt.title("Top 10 Important Features")
        st.pyplot(plt)
    except:
        st.info("Feature importance not available.")
