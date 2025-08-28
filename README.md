**Risk Evaluation & Underwriting System**

This project is a Flask-based Machine Learning web application that predicts whether a loan should be approved or denied based on applicant financial details. 
It enhances risk assessment and loan underwriting by using a trained ML model and probability thresholds to make decisions.

**Features**

🧾 User Input Form: Collects borrower details such as loan amount, income, DTI, credit score, etc.

🧠 ML-Powered Predictions: Uses a trained model (final_best_model.pkl) to classify loan applications.

📊 Probability & Risk Score: Displays probability of loan approval and associated risk level.

✅ Approval/Denial Decision: Returns either Loan Approved or Loan Denied based on prediction.

🔄 Data Scaling: Uses a pre-trained scaler.pkl to normalize input features before prediction.

**Technologies Used**

Python 3.x

Flask → Web framework

Pandas → Data manipulation

Scikit-learn → ML model & preprocessing

Joblib → Model persistence

HTML/CSS  → Frontend
