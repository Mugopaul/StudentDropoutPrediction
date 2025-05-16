import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Loading the model
model=pickle.load(open('student_dropout_model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output)
    label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    prediction_label = label_map.get(int(output), "Unknown")
    return jsonify({'prediction': prediction_label})
    ##return jsonify({'prediction': output[0].item()})
    ##return jsonify(output[0])
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    label_map={0: 'Dropout', 1:'Enrolled', 2: 'Graduate'}
    prediction_label = label_map.get(int(output), "Unknown")
    return render_template("home.html",prediction_text=f"The Student is likely to {prediction_label}")
 
 

if __name__=="__main__":
    app.run(debug=True)
