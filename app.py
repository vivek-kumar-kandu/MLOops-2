from flask import Flask , render_template,request
import joblib

app=Flask(__name__)

model=joblib.load("model/loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Age= float(request.form["Age"])
    
    Income= float(request.form["Income"])
    
    Loan_Amount= float(request.form["Loan_Amount"])
    
    Credit_Score= float(request.form["Credit_Score"])
    
    Loan_Status = model.predict([[Age,Income,Loan_Amount,Credit_Score]])
    
    if Loan_Status[0]==1:
        result="Loan Approved"
    else:
        result="Loan Rejected"
    
    return render_template("index.html", prediction_text=result, Age=Age, Income=Income,  Loan_Amount=Loan_Amount, Credit_Score=Credit_Score)

if __name__ =="__main__":
    app.run(host='0.0.0.0',port=8000)
