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
        file_path='voice_recognition/static/assets/recordedAudio.wav'
        file.save(os.path.join(file_path))
        mostafa_score,magdy_score,mayar_score,mina_score,open_score=comparing(file_path)
        print(mostafa_score)
        print(magdy_score)
        print(mayar_score)
        print(mina_score)
        print(open_score)
        name = ""
        if mostafa_score == max(mostafa_score,magdy_score,mayar_score,mina_score,open_score):
            name = "Mostafa"
        elif mina_score == max(mostafa_score,magdy_score,mayar_score,mina_score,open_score):
            name = "Mina"
        elif magdy_score == max(mostafa_score,magdy_score,mayar_score,mina_score,open_score):
            name = "Magdy"
        elif mayar_score == max(mostafa_score,magdy_score,mayar_score,mina_score,open_score):
            name = "Mayar"
        elif open_score == max(mostafa_score,magdy_score,mayar_score,mina_score,open_score):
            name = "other"

    return f'<h1 id="statement">Hello {name}</h1>'

if __name__ == '__main__':
    app.run(debug=True)