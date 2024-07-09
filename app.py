from flask import Flask, jsonify, request
import os
from Logistic_iris import MachineLearningModel

machine_learning_obj = MachineLearningModel()
app = Flask(__name__)

@app.route('/predict_flower', methods=['POST'])
def predict_flower():
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    model_path = 'File/logistic_regression_model.pickle'

    if os.path.exists(model_path):
        machine_learning_obj.load(model_path)
    else:
        machine_learning_obj.train()

    y_pred = machine_learning_obj.test(sepal_length, sepal_width, petal_length, petal_width)
    return jsonify({'result': y_pred})

if __name__ == '__main__':
    app.run(debug=True)

