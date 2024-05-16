from flask import Flask,render_template,request
import joblib
import os
import numpy as np
import pickle
import sklearn

app=Flask(__name__)
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])

# GET is used to request data from a specified resource.
# POST is used to send data to a server to create/update a resource.

def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease=int(request.form['heart_disease'])
    ever_married=int(request.form['ever_married'])
    work_type=int(request.form['work_type'])
    Residence_type=int(request.form['Residence_type'])
    avg_glucose_level=int(request.form['avg_glucose_level'])
    bmi=int(request.form['bmi'])
    smoking_status=int(request.form['smoking_status'])
    # take input data into array
    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level
                ,bmi,smoking_status]).reshape(1,-1)
    # now we need to scale down data

    scaler_path=os.path.join(r'C:\Users\yathi\campusx_ml\resume_project_ml','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)
    dummy=os.path.join(r'C:\Users\yathi\campusx_ml\resume_project_ml','models/rf.sav')
    rf=joblib.load(dummy)
    r=rf.predict(x)

    # for no stroke problem
    if r==0:
        return render_template("nostroke.html")
    else:
        return render_template("stroke.html")

if __name__=="__main__":
    app.run(debug=True,port=1234)
