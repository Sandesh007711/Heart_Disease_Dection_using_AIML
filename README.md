# Heart Disease Detection using AI/ML

This project leverages Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict the likelihood of heart disease in patients based on clinical data. The goal is to assist healthcare professionals in identifying potential heart disease cases earlier, improving the chances of timely intervention.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Heart disease is a leading cause of mortality globally. Early detection is essential for effective treatment. This project implements various ML algorithms to analyze patient data and predict the presence of heart disease.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Implementation of multiple ML models (e.g., Logistic Regression, Random Forest, SVM, KNN)
- Model evaluation and comparison
- Predictive web interface (optional)
- Visualization of important features

## Dataset

The project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) or any similar open-source dataset. The dataset typically includes features like:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Target (diagnosis of heart disease)

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- (Optional) Flask/Streamlit for web deployment

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sandesh007711/Heart_Disease_Dection_using_AIML.git
   cd Heart_Disease_Dection_using_AIML
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main notebook or script:**
   - Open `Heart_Disease_Prediction.ipynb` in Jupyter Notebook
   - Or run the Python script if provided:
     ```bash
     python main.py
     ```

4. **(Optional) Run the web app:**
   ```bash
   streamlit run app.py
   # or
   flask run
   ```

## Project Structure

```
Heart_Disease_Dection_using_AIML/
├── data/
│   └── heart.csv
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Results

- Achieved high accuracy in predicting heart disease using selected ML models.
- Feature importance identified for better interpretability.
- Visualizations provide insights into data and model performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions and improvements.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Disclaimer:** This project is intended for educational purposes only and should not be used for actual medical diagnosis.
