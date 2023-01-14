
from flask import Flask, request, render_template
import os
import cv2
import pickle as pkl


app = Flask(__name__)


def processImg(IMG_PATH):
    # загрузка модели
    with open("model.pkl", "rb") as f:
        model = pkl.load(f)
    
    # чтение и предобработка изображения
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (28,28))[:,:,1]
    image = image.flatten()

    # определение символа
    predictions = model.predict(image.reshape(1,-1))

    return list(map(int, list(predictions)))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['POST'])
def upload_file():
    predictions=""

    if request.method == 'POST':
        f = request.files['file']
        f.save("static/img.jpg")    
        preds = processImg("static/img.jpg")
    
    return render_template("upload.html", predictions=preds, display_image="../img.jpg") 


if __name__ == "__main__":
    app.run()
