from flask import Flask,render_template,redirect,request
import pickle
import pandas as pd
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        model=pickle.load(open('svc_model.pkl','rb'))
        variance=float(request.form['variance'])
        skewness=float(request.form['skewness'])
        curtosis=float(request.form['curtosis'])
        entropy=float(request.form['entropy'])
        scaler=pickle.load(open('scaler.pkl','rb'))
        x=[[variance,skewness,curtosis,entropy]]
        x=scaler.transform(x)
        output=model.predict(pd.DataFrame(x))[0]
        if int(output)==0:
            result='You are not authenticated'
        else:
            result='You are authenticated'    
        return render_template('index.html',prediction=result)
    return render_template('index.html')    

if __name__=='__main__':
    app.run(debug=True)    