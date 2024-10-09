from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('final_best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        loan_amnt = float(request.form['loan_amnt'])
        int_rate = float(request.form['int_rate'])
        annual_inc = float(request.form['annual_inc'])
        dti = float(request.form['dti'])
        fico_range_low = float(request.form['fico_range_low'])
        open_acc = int(request.form['open_acc'])
        revol_bal = float(request.form['revol_bal'])
        revol_util = float(request.form['revol_util'])
        delinq_2yrs = int(request.form['delinq_2yrs'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[loan_amnt, int_rate, annual_inc, dti,
                                     fico_range_low, open_acc, revol_bal,
                                     revol_util, delinq_2yrs]],
                                   columns=['loan_amnt', 'int_rate', 'annual_inc', 
                                            'dti', 'fico_range_low', 'open_acc', 
                                            'revol_bal', 'revol_util', 'delinq_2yrs'])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make a prediction
        y_proba = model.predict_proba(input_scaled)[:, 1]
        threshold = 0.7
        prediction = (y_proba >= threshold).astype(int)
        risk_level = 1 - y_proba[0]

        # Convert the prediction to a more understandable format
        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Denied"

        # Pass form data back to the template
        return render_template('index.html', 
                               prediction=result, 
                               probability=y_proba[0], 
                               risk_level=risk_level,
                               loan_amnt=loan_amnt, int_rate=int_rate, annual_inc=annual_inc,
                               dti=dti, fico_range_low=fico_range_low, open_acc=open_acc,
                               revol_bal=revol_bal, revol_util=revol_util, delinq_2yrs=delinq_2yrs)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True)