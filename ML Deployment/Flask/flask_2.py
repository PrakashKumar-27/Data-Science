from flask import Flask, render_template, request

app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template('index.html');

@app.route('/form')
def form():
    return render_template('form.html');

@app.route('/predict', methods=["POST"])
def predict():
    exp = request.form.get('exp');
    score = request.form.get('score');
    points = request.form.get('points');
    print(exp , score, points)
    return "Form submitted";

app.run(debug=True)