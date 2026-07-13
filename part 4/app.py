"""
app.py - Heart Disease Risk Checker (SIMPLE version)
========================================================
HOW TO RUN THIS:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in your browser.

"""

import math
from flask import Flask, request, render_template_string

app = Flask(__name__)


# =====================================================================
# THE TRAINED MODEL, AS PLAIN NUMBERS
# =====================================================================
# These numbers came from training a logistic regression model on 650
# patient records. A logistic regression model works like this:
#   1. Scale each input:      scaled = (value - mean) / spread
#   2. Multiply each scaled input by its own "weight" (coefficient)
#   3. Add all of those up, plus one starting number (the intercept)
#   4. Squash that total through a sigmoid curve to get a 0-1 probability
# That's ALL a logistic regression model does -- no black box, just
# the arithmetic below.

FEATURES = [
    #   name                 mean       spread(std)   coefficient (weight)
    ("Age",                 54.66,      20.59,        0.702),
    ("BMI",                 27.20,       5.37,        0.471),
    ("SystolicBP",         125.42,      15.68,        0.607),
    ("DiastolicBP",         79.74,      10.33,       -0.047),
    ("GlucoseMgDl",         99.42,      19.59,       -0.165),
    ("RestingHR",           72.06,      11.18,        0.070),
    ("ExerciseHrsPerWeek",   2.27,       2.21,       -0.079),
    ("Sex_Male",             0.50,       0.50,       -0.057),
    ("Smoker_Yes",           0.22,       0.42,        0.657),
    ("FamilyHistory_Yes",    0.32,       0.46,        0.486),
    ("Diabetes_Yes",         0.14,       0.35,        0.268),
]
INTERCEPT = -0.175


def predict_risk(inputs: dict) -> float:
    """
    Takes a dictionary of patient values (e.g. {'Age': 71, 'BMI': 33.2, ...})
    and returns a probability between 0 and 1, using the model numbers above.
    """
    total = INTERCEPT
    for name, mean, spread, weight in FEATURES:
        value = inputs[name]
        scaled_value = (value - mean) / spread
        total += scaled_value * weight

    probability = 1 / (1 + math.exp(-total))  # the sigmoid function
    return probability


# =====================================================================
# THE WEBPAGE (HTML lives right here as a string -- no separate files)
# =====================================================================

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Heart Disease Risk Checker</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; color: #222; }
    h1 { font-size: 24px; }
    .field { margin-bottom: 14px; }
    label { display: block; font-weight: bold; margin-bottom: 4px; font-size: 14px; }
    input, select { width: 100%; padding: 8px; font-size: 14px; box-sizing: border-box; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
    .checkbox-row label { font-weight: normal; margin: 0; }
    button { background: #2c7a4b; color: white; border: none; padding: 12px 22px; font-size: 15px; border-radius: 4px; cursor: pointer; margin-top: 14px; }
    button:hover { background: #24623c; }
    .result { margin-top: 24px; padding: 18px; border-radius: 6px; }
    .result.low { background: #e6f4ea; border: 1px solid #2c7a4b; }
    .result.medium { background: #fff4e0; border: 1px solid #b8860b; }
    .result.high { background: #fbe9e7; border: 1px solid #c0392b; }
    .disclaimer { font-size: 12px; color: #666; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>&#10084;&#65039; Heart Disease Risk Checker</h1>
  <p>Fill in the fields and click "Check risk."</p>

  <form method="POST" action="/">
    <div class="two-col">
      <div class="field">
        <label>Age (years)</label>
        <input type="number" name="Age" value="50" required>
      </div>
      <div class="field">
        <label>Sex</label>
        <select name="Sex">
          <option value="Female">Female</option>
          <option value="Male">Male</option>
        </select>
      </div>
      <div class="field">
        <label>BMI</label>
        <input type="number" step="0.1" name="BMI" value="26" required>
      </div>
      <div class="field">
        <label>Systolic BP (mmHg)</label>
        <input type="number" name="SystolicBP" value="122" required>
      </div>
      <div class="field">
        <label>Diastolic BP (mmHg)</label>
        <input type="number" name="DiastolicBP" value="78" required>
      </div>
      <div class="field">
        <label>Glucose (mg/dL)</label>
        <input type="number" name="GlucoseMgDl" value="99" required>
      </div>
      <div class="field">
        <label>Resting heart rate (bpm)</label>
        <input type="number" name="RestingHR" value="72" required>
      </div>
      <div class="field">
        <label>Exercise (hrs/week)</label>
        <input type="number" step="0.5" name="ExerciseHrsPerWeek" value="2" required>
      </div>
    </div>

    <div class="field">
      <div class="checkbox-row"><input type="checkbox" name="Smoker" id="smoker"><label for="smoker">Currently smokes</label></div>
      <div class="checkbox-row"><input type="checkbox" name="Family" id="family"><label for="family">Family history of heart disease</label></div>
      <div class="checkbox-row"><input type="checkbox" name="Diabetes" id="diabetes"><label for="diabetes">Diagnosed with diabetes</label></div>
    </div>

    <button type="submit">Check risk</button>
  </form>

  {% if result %}
  <div class="result {{ result.band_class }}">
    <h2>{{ result.pct }}% estimated probability</h2>
    <p><strong>Risk band:</strong> {{ result.band }}</p>
  </div>
  {% endif %}

  <p class="disclaimer">Demonstration tool built on a synthetic dataset. Not a diagnostic device.</p>
</body>
</html>
"""


# =====================================================================
# THE ONE ROUTE -- shows the form, and handles the form being submitted
# =====================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        inputs = {
            "Age": float(request.form['Age']),
            "BMI": float(request.form['BMI']),
            "SystolicBP": float(request.form['SystolicBP']),
            "DiastolicBP": float(request.form['DiastolicBP']),
            "GlucoseMgDl": float(request.form['GlucoseMgDl']),
            "RestingHR": float(request.form['RestingHR']),
            "ExerciseHrsPerWeek": float(request.form['ExerciseHrsPerWeek']),
            "Sex_Male": 1 if request.form.get('Sex') == 'Male' else 0,
            "Smoker_Yes": 1 if 'Smoker' in request.form else 0,
            "FamilyHistory_Yes": 1 if 'Family' in request.form else 0,
            "Diabetes_Yes": 1 if 'Diabetes' in request.form else 0,
        }

        probability = predict_risk(inputs)
        pct = round(probability * 100)

        if probability < 0.34:
            band, band_class = "LOW", "low"
        elif probability < 0.66:
            band, band_class = "MEDIUM", "medium"
        else:
            band, band_class = "HIGH", "high"

        result = {"pct": pct, "band": band, "band_class": band_class}

    return render_template_string(PAGE, result=result)


if __name__ == '__main__':
    app.run(debug=True)
