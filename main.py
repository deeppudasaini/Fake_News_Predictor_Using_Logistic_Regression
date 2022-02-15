from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('logistic_regression.pkl','rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        title=str(request.form['title'])
        author=str(request.form['author'])
        titleAuthor=str(title+' '+author)
        inputData=np.asarray(titleAuthor)
        final_input=inputData.reshape(1,-1)
        prediction = model.predict(final_input)
        
        if prediction[0] == 0:
            return render_template('index.html', prediction_texts="The news provided is fake news")
        elif prediction[0] == 1:
            return render_template('index.html', prediction_texts="The news provided is real news")

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)