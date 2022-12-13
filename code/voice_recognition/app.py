from flask import Flask, render_template, send_file, request, redirect
from routes import *
import os
from wtforms.validators import InputRequired
import librosa

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
        mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score=comparing(file_path)
        print(mostafa_score)
        print(magdy_score)
        print(mayar_score)
        print(mina_score)
        print(others_score)
        print(close_score)
        name = ""
        if mostafa_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Correct Voice Fingerprint, Mostafa"
        elif mina_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Correct Voice Fingerprint, Mina"
        elif magdy_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Correct Voice Fingerprint, Magdy"
        elif mayar_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Correct Voice Fingerprint, Mayar"
        elif others_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Wrong Voice Fingerprint!"
        elif close_score == max(mostafa_score,magdy_score,mayar_score,mina_score,others_score,close_score):
            name = "Wrong Voice Fingerprint!"

    return f'<h1 id="statement">{name}</h1>'

if __name__ == '__main__':
    app.run(debug=True)