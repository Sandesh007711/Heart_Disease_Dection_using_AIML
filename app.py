from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])  # Chest pain type
            trestbps = int(request.form['trestbps'])  # Resting blood pressure
            chol = int(request.form['chol'])  # Serum cholesterol
            fbs = int(request.form['fbs'])  # Fasting blood sugar
            restecg = int(request.form['restecg'])  # Resting electrocardiographic results
            thalach = int(request.form['thalach'])  # Maximum heart rate achieved
            exang = int(request.form['exang'])  # Exercise induced angina
            oldpeak = float(request.form['oldpeak'])  # ST depression induced by exercise
            slope = int(request.form['slope'])  # Slope of peak exercise ST segment
            ca = int(request.form['ca'])  # Number of major vessels colored by fluoroscopy
            thal = int(request.form['thal'])  # Thalassemia

            # Prepare features for prediction
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Make the prediction
            prediction = model.predict(features)
            
            # Display the result on the webpage
            if prediction[0] == 1:
                result = "Heart disease is present."
            else:
                result = "Heart disease is not present."
            
            return render_template('index.html', prediction=result)

        except KeyError as e:
            return f"Error: Missing key {e}"
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
