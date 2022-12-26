import numpy as np
import sys
import pickle
from flask import Flask, request, render_template
from flask_cors import CORS


app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/': {"origins": "http://localhost:8082"}})

try:
    model = pickle.load(open('../../experiments/log_reg.sav', "rb"))
except FileNotFoundError:
    sys.exit(1)


@app.route('/', methods=['GET', 'POST'])
def get_wine_quality():
    # 10.8,0.47,0.43,2.1,0.171,27.0,66.0,0.9982,3.17,0.76,10.8
    x = request.form.get('winedata')
    if request.method == 'POST':
        x = np.array(x.split(','), dtype=np.float32).reshape(1, -1)
        y = model.predict(x)[0]
        return f'wine quality is: {y}'
    return render_template('form.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    