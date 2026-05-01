# 🫀 ECG Arrhythmia Detection using Machine Learning

##  Overview
This project implements a machine learning pipeline to detect **abnormal heart rhythms (arrhythmias)** using ECG-derived features.

Multiple machine learning models are trained and compared to classify heartbeats as:
- **Normal (0)**
- **Abnormal (1)**

The system serves as a **decision-support prototype** for early detection of cardiac abnormalities, which can assist in prioritizing high-risk cases in healthcare settings.

---

##  Problem Statement
Cardiac arrhythmias can lead to severe conditions such as stroke, cardiac arrest, and sudden death if not detected early.

Manual ECG interpretation:
- Requires expert knowledge  
- Is time-consuming  
- Is prone to human variability  

This project aims to:
- Automate ECG-based arrhythmia detection  
- Improve early screening efficiency  
- Support clinical decision-making  

---

##  Approach

### 1. Data Preprocessing
- Used MIT-BIH ECG feature dataset  
- Converted multi-class labels into binary classification:
  - `0 → Normal`
  - `1 → Abnormal`
- Removed unnecessary columns
- Handled missing values
- Applied feature scaling (StandardScaler)

---

### 2. Machine Learning Models
The following models were trained and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

---

### 3. Evaluation Metrics
Models were evaluated using:
- F1 Score (primary metric)
- ROC-AUC Score
- Confusion Matrix



##  Results

- Multiple ML models were compared  
- Best model selected based on performance metrics  
- Random Forest provided strong interpretability  

## Model Selection Strategy

Although Random Forest achieved strong overall performance, 
Support Vector Machine (SVM) was selected as the final model 
because it demonstrated better recall for detecting abnormal heartbeats.

In healthcare applications, minimizing false negatives is critical, 
as missing an abnormal case can have serious consequences.

##  Visualizations

The project includes:
-  Model Comparison Graph  
-  Confusion Matrix  
-  Feature Importance Plot  

These help in understanding both performance and interpretability.

## Feature Importance 

Important ECG features identified:
- RR intervals  
- QRS-related features  
- Morphological signal characteristics  

These align with known physiological indicators of heart rhythm abnormalities.

---

## Project Structure
arrhythmia-ml-app/
│
├── models/
│ ├── svm_model.pkl
│ ├── scaler.pkl
│
├── results/
│ ├── model_comparison.png
│ ├── confusion_matrix.png
│ └── feature_importance.png
│
├── train.py
├── requirements.txt
└── README.md

## Note: 
Trained models are not included due to size constraints. 
You can generate them by running the script.

## Key Insights
- Ensemble models (Random Forest) performed best for this dataset
- Feature scaling improves performance for distance-based models
- Feature importance adds interpretability to ML predictions
- Model comparison helps select robust classifiers


⚠️ Disclaimer

This project is intended for educational and research purposes only.
It is not a medical diagnostic tool and should not be used for clinical decisions.

## Author
Kashaf Raheem
AI/Software Engineer