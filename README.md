# Draw a digit

"Draw a digit" is a simple Flask web app for **mouse-or-finger-written digit recognition**.

Demo:[site](https://)

<p align="center">
  <img width="1200" src="/static/images/screenshot_home.png">
</p>


## Installation
Make sure you have Python3.7+ on your machine or virtual environment. 

The app should work with Python3.6 as well but I couldn't test it because XCode 12.4 conflicts with pyenv; raise an issue if you encounter any problem.

It is recommended to upgrade pip:

    pip install --upgrade pip
    
then just run:

    pip install -r requirements.txt
   
## Usage

Run the app:

    python3 app.py
    
The app will run on: [http://localhost:8080](http://localhost:8080/)

---

The model is already trained but if you want to modify something and retrain it, you just have to launch:

    python3 model/CNN.py
    
If you want to run cross-validation and make a plot of the performance, you have to uncomment the lines:

    # histories = evaluate_model(trainX, trainY)
    # model_performance(histories)

## Model

The model is a 2xCNN layers with the following architecture:

![CNN architecture](/static/images/nn.png)

For more details see ``model\CNN.py``.

The cross-validation accuracy of the model is **99.5%**.
    
