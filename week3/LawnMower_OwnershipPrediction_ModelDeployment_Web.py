from flask import Flask, render_template, request

import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
LawnMower_Model = pickle.load(open('./data/svm_lin_model.pkl', "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        income = float(request.form["income"])
        lot_size = float(request.form["lot_size"])
        df = pd.DataFrame({'Income': [income], 'Lot_Size': [lot_size]})
        result = LawnMower_Model.predict(df)
        ownership = ('Potential lawn mower owner', 'Not a potential lawn mower owner')
        return render_template('result.html', ownership=ownership[result[0]])

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
