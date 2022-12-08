from flask import Flask, render_template, send_file, request, redirect
from routes import *
import os
from wtforms.validators import InputRequired
import librosa
# import matplotlib.pyplot as plt
# import numpy as np




app = Flask(__name__)


@app.route('/', methods=['GET', "POST", "put"])
def index():
    # if request.method == "POST":
    return render_template('index.html')



@app.route('/saveRecord',methods =['POST'])
def save_record():
    if request.method =='POST':
        file=request.files['AudioFile']
        file_path='static/assets/recordedAudio.wav'
        file.save(os.path.join(file_path))
        mostafa_score,magdy_score,mayar_score,mina_score=comparing(file_path)
        print(mostafa_score)
        print(magdy_score)
        print(mayar_score)
        print(mina_score)
        name = ""
        if mostafa_score == max(mostafa_score,magdy_score,mayar_score,mina_score):
            name = "mostafa"
        elif mina_score == max(mostafa_score,magdy_score,mayar_score,mina_score):
            name = "mina"
        elif magdy_score == max(mostafa_score,magdy_score,mayar_score,mina_score):
            name = "magdy"
        elif mayar_score == max(mostafa_score,magdy_score,mayar_score,mina_score):
            name = "mayar"

    return f'<h1>success {name}</h1>'

if __name__ == '__main__':
    app.run(debug=True)