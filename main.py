import json
import numpy as np

from flask import Flask, render_template, request, redirect
#import tensorflow as tf

from PIL import Image, ImageOps, ImageChops
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
#app.config['FONTAWESOME_STYLES'] = ['brands', 'solid']
#fa = FontAwesome(app)
model = None


# def model_loader():
#     global model
#     model = load_model('32C5-P2_64C5-P2-128.h5')


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


# def prepare_image(img):
#     img = ImageOps.invert(img)
#     img = center_image(img)
#     img = img.resize((28, 28))
#     img = img_to_array(img)
#     img = img.reshape(1, 28, 28, 1)
#     img = img.astype('float32')
#     img = (img / 255.0) - 0.5
#     return img


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict_digit():
    res_json = {}
##    raw_img = Image.open(request.files['img']).convert('L')
##    img = prepare_image(raw_img)

    # model_loader()
##    predictions = model.predict(img)
##    probs = predictions[0]*100
##    pred = str(np.argmax(predictions))
##    res_json['probs'] = probs.tolist()
##    res_json['pred'] = pred
    res_json['pred'] = 5
    res_json['probs'] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    return json.dumps(res_json)


if __name__ == '__main__':
    print(("* Loading model, please wait..."))
# model_loader()  # model should always be loaded here, once
    app.run(host='127.0.0.1', port=8080, debug=True)
