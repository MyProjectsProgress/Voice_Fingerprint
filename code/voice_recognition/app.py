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
        scores_mostafa,scores_magdy,scores_mayar,scores_mina=comparing(file_path)
        print("Mostafa", scores_mostafa)
        print("Magdy",scores_magdy)
        print("Mayar",scores_mayar)
        print("Mina",scores_mina)
        
        max_score = max(scores_mostafa,scores_magdy,scores_mayar,scores_mina)
        name = ""
        if(max_score < -30):
            name = "Wrong Voice Fingerprint!"
        else:
            if scores_mostafa == max_score:
                name = "Correct Voice Fingerprint, Mostafa"
            elif scores_magdy == max_score:
                name = "Correct Voice Fingerprint, Magdy"
            elif scores_mayar == max_score:
                name = "Correct Voice Fingerprint, Mayar"
            elif scores_mina == max_score:
                name = "Correct Voice Fingerprint, Mina"

    return f'<h1 id="statement">{name}</h1>'

if __name__ == '__main__':
    app.run(debug=True)