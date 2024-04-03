from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('stacking_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='you need to go for a treatment.\nProbability of mental health is  {}'.format(output),bhai="feel free to have a treatment")
    else:
        return render_template('index.html',pred='you have a good mental health state.\n Probability of mental health is {}'.format(output),bhai="good mental state")


if __name__ == '__main__':
    app.run(debug=True)