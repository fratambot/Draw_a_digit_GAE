import json
import os
import numpy as np

from flask import Flask, render_template, request

from PIL import Image, ImageOps, ImageChops
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

app = Flask(__name__)


MODEL = load_model("32C5-P2_64C5-P2-128.h5")


def center_image(img):
    w, h = img.size
    left, top, right, bottom = w, h, -1, -1
    imgpix = img.getdata()

    for y in range(h):
        offset_y = y * w
        for x in range(w):
            if imgpix[offset_y + x] > 0:  # exclude the origin
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    shiftX = (left + (right - left) // 2) - w // 2
    shiftY = (top + (bottom - top) // 2) - h // 2

    return ImageChops.offset(img, -shiftX, -shiftY)


def prepare_image(img):
    img = ImageOps.invert(img)
    img = center_image(img)
    img = img.resize((28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = (img / 255.0) - 0.5
    return img


@app.route('/')
def root():
    return render_template('index.html')


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("images/favicon.ico")


@app.route('/prediction', methods=['GET', 'POST'])
def predict_digit():
    res_json = {}
    raw_img = Image.open(request.files['img']).convert('L')
    img = prepare_image(raw_img)
    if MODEL is not None:
        predictions = MODEL.predict(img)
        probs = predictions[0]*100
        pred = str(np.argmax(predictions))
        res_json['probs'] = probs.tolist()
        res_json['pred'] = pred

    return json.dumps(res_json)


#assert os.path.exists(MODEL_PATH), "no saved model"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
