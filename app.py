from flask import Flask,render_template,request
import pickle
import json
import numpy as np

model = pickle.load(open("model.pkl","rb"))

tfidf = pickle.load(open("tfidf_vec.pkl","rb"))

encoder = pickle.load(open("encoder.pkl","rb"))

with open("columns_name.json","r") as json_file:
    col_name = json.load(json_file)
col_name_list = col_name['col_name']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods= ["GET","POST"])
def predict():
    data = request.form["text"]
    input_data = np.zeros(len(col_name_list))

    Text = ["".join(data)]
    input_data = tfidf.transform(Text).toarray()

    print(input_data)
    my_prediction = model.predict(input_data)
    class_list  = encoder.classes_
    result = class_list[my_prediction[0]]
    print(result)
    
    return render_template("index.html",prediction = result)

if __name__== "__main__":
    app.run(host= "0.0.0.0",port=8080, debug=True)