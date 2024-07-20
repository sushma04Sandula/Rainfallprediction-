from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as file:
    random_forest = pickle.load(file)

# Load data and preprocess
data = pd.read_csv("rainfall.csv")

# Fill NaN values for numeric columns only
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Prepare data for modeling
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            state = request.form["state"]
            month = request.form["month"]
            year = int(request.form["year"])

            # Check if the selected year is beyond the last year in historical data (2015)
            if year > 2015:
                # Predict future rainfall
                pred_data = prepare_pred_data(state, month, year)
                res = predict_rainfall(pred_data)
                return render_template('result.html', res=res, state=state)

            # Get monthly rainfall data for the selected state and year
            state_data = data[(data['SUBDIVISION'] == state) & (data['YEAR'] == year)]

            if len(state_data) == 0:
                return render_template('index.html', error_message=f"No data found for {state} in {year}")

            # Aggregate the rainfall data for the selected month period
            if month == "JAN-FEB":
                rainfall = state_data[['JAN', 'FEB']].sum(axis=1).values[0]
            elif month == "MAR-APR":
                rainfall = state_data[['MAR', 'APR']].sum(axis=1).values[0]
            elif month == "MAY-JUN":
                rainfall = state_data[['MAY', 'JUN']].sum(axis=1).values[0]
            elif month == "JUL-AUG":
                rainfall = state_data[['JUL', 'AUG']].sum(axis=1).values[0]
            elif month == "SEP-OCT":
                rainfall = state_data[['SEP', 'OCT']].sum(axis=1).values[0]
            elif month == "NOV-DEC":
                rainfall = state_data[['NOV', 'DEC']].sum(axis=1).values[0]

            return render_template('result.html', res=rainfall, state=state)

        except KeyError as e:
            return f"Missing form field: {e}"
        except ValueError as e:
            return f"Value error: {e}"

    # Construct image URLs
    state_images = {
        "ANDAMAN & NICOBAR ISLANDS": url_for('static', filename='images/andamannicobar.jpg'),
        "ARUNACHAL PRADESH": url_for('static', filename='images/arunachalpradesh.jpg'),
        "ASSAM": url_for('static', filename='images/assam.jpg'),
        "BIHAR": url_for('static', filename='images/bihar.jpg'),
        "CHHATTISGARH": url_for('static', filename='images/chattisgarh.jpg'),
        "GOA": url_for('static', filename='images/goa.jpg'),
        "GUJARAT": url_for('static', filename='images/gujarat.jpg'),
        "HARYANA": url_for('static', filename='images/haryana.jpg'),
        "HIMACHAL PRADESH": url_for('static', filename='images/himachal.jpg'),
        "JAMMU & KASHMIR": url_for('static', filename='images/jammukashmir.jpg'),
        "JHARKHAND": url_for('static', filename='images/jharkand.jpg'),
        "KARNATAKA": url_for('static', filename='images/karnataka.jpg'),
        "KERALA": url_for('static', filename='images/kerala.jpg'),
        "MADHYA PRADESH": url_for('static', filename='images/madyapradesh.jpg'),
        "MAHARASHTRA": url_for('static', filename='images/maharashtra.jpg'),
        "MANIPUR": url_for('static', filename='images/manipur.jpg'),
        "MEGHALAYA": url_for('static', filename='images/meghalaya.jpg'),
        "MIZORAM": url_for('static', filename='images/mizoram.jpg'),
        "NAGALAND": url_for('static', filename='images/nagaland.jpg'),
        "ODISHA": url_for('static', filename='images/odisha.jpg'),
        "PUNJAB": url_for('static', filename='images/punjab.jpg'),
        "RAJASTHAN": url_for('static', filename='images/rajasthan.jpg'),
        "SIKKIM": url_for('static', filename='images/sikkim.jpg'),
        "TAMIL NADU": url_for('static', filename='images/tamilnadu.jpg'),
        "TELANGANA": url_for('static', filename='images/telangana.jpg'),
        "TRIPURA": url_for('static', filename='images/tripura.jpg'),
        "UTTAR PRADESH": url_for('static', filename='images/uttarpradesh.jpg'),
        "UTTARAKHAND": url_for('static', filename='images/uttarakand.jpg'),
        "WEST BENGAL": url_for('static', filename='images/westbengal.jpg')
    }

    return render_template('index.html', state_images=state_images)


def prepare_pred_data(state, month, year):
    # Prepare data for prediction
    pred_data = pd.DataFrame({
        'SUBDIVISION': [state],
        'YEAR': [year],
        'RAINFALL': [np.nan],  # Placeholder for future year
        'JAN': [1 if month == 'JAN-FEB' else 0],
        'FEB': [1 if month == 'JAN-FEB' else 0],
        'MAR': [1 if month == 'MAR-APR' else 0],
        'APR': [1 if month == 'MAR-APR' else 0],
        'MAY': [1 if month == 'MAY-JUN' else 0],
        'JUN': [1 if month == 'MAY-JUN' else 0],
        'JUL': [1 if month == 'JUL-AUG' else 0],
        'AUG': [1 if month == 'JUL-AUG' else 0],
        'SEP': [1 if month == 'SEP-OCT' else 0],
        'OCT': [1 if month == 'SEP-OCT' else 0],
        'NOV': [1 if month == 'NOV-DEC' else 0],
        'DEC': [1 if month == 'NOV-DEC' else 0]
    })

    return pred_data


def predict_rainfall(pred_data):
    # Ensure categorical columns are encoded properly
    label_encoders = {}
    for feature in ['SUBDIVISION', 'YEAR']:
        label_encoders[feature] = LabelEncoder()
        pred_data[feature] = label_encoders[feature].fit_transform(pred_data[feature])

    # Ensure the input has exactly 12 features
    required_features = ['SUBDIVISION', 'YEAR', 'RAINFALL', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    pred_data = pred_data[required_features]

    # Predict using the model
    pred_data_array = pred_data.values.reshape(1, -1)[:, :12]  # Select only the first 12 features
    res = random_forest.predict(pred_data_array)[0]
    res = round(res, 2)

    return res


if __name__ == "__main__":
    app.run(debug=True)