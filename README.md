🚧 Accident Prediction With Pipeline

This Streamlit web app predicts the severity of road accidents based on multiple features using a trained Random Forest Classifier. The model pipeline includes preprocessing, encoding, handling imbalance using RandomOverSampler, and model deployment.

📌 Features
Full pipeline: preprocessing → encoding → balancing → model training

Trained with RandomForestClassifier

Built with scikit-learn, imbalanced-learn, joblib, and Streamlit

Simple and interactive UI to upload input and predict accident severity

📁 File Structure
bash
Copy
Edit
├── 10 pipe dataset.csv         # Dataset used for training
├── 10 pipe img.png             # Pipeline diagram
├── Pipeline.py                 # Code to create the ML pipeline and save the model
├── app.py                      # Streamlit app to load model and make predictions
├── requirements.txt            # List of required packages
└── README.md                   # Project documentation

▶️ How to Run the Project

1. Clone the Repository

bash
git clone https://github.com/mukundkhandelwal463/Accident-Prediction-With-Pipeline.git
cd Accident-Prediction-With-Pipeline

3. Install Dependencies
bash
pip install -r requirements.txt

5. Run the Streamlit App
bash
streamlit run app.py

🔍 Sample Dataset
The dataset contains multiple features relevant to road accidents. It was cleaned and used to build a supervised classification model. The pipeline handles:
Missing values
Categorical encoding
Balancing using RandomOverSampler

🔧 Technologies Used
Python
Pandas, NumPy
Scikit-learn
imbalanced-learn
Matplotlib, Seaborn
Streamlit
Joblib

📊 Pipeline Diagram

📬 Contact
Made with ❤️ by Mukund Khandelwal

Project Link: https://github.com/mukundkhandelwal463/mukundkhandelwal463-Accident-Prediction-With-Pipeline/tree/main

Linkedin: https://www.linkedin.com/posts/mukund-khandelwal-6a8663283_machinelearning-datascience-python-activity-7356009127363403779-rFyr?utm_source=share&utm_medium=member_desktop&rcm=ACoAAET5diABs7bbZlDnVTGZ4DnPgeKxnEmHsgA

