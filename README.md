#  Student Performance Predictor — Random Forest From Scratch

A machine learning portfolio project that predicts a student's final exam score using a **Random Forest Regressor implemented from scratch**.

##  What makes this project different?

The Random Forest algorithm is **not imported from scikit-learn**.

The project implements:

- Decision trees from scratch
- Bootstrap sampling
- Random feature selection
- Mean Squared Error splitting
- Recursive tree construction
- Multiple decision trees
- Ensemble averaging
- Feature importance

Only Python, NumPy, and Pandas are used for the machine learning implementation.

## Features

The model uses:

- Study hours per day
- Attendance percentage
- Previous exam score
- Assignments completed
- Sleep hours
- Extracurricular activities

## Machine Learning Pipeline

```text
Dataset
   ↓
Manual Train/Test Split
   ↓
Bootstrap Sampling
   ↓
Random Feature Selection
   ↓
Decision Tree
   ↓
Repeat for Multiple Trees
   ↓
Average Tree Predictions
   ↓
Final Score
```

## Project Structure

```text
student-performance-random-forest-from-scratch/
│
├── data/
│   └── student_data.csv
│
├── model/
│   └── random_forest.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── random_forest.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Clone

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd student-performance-random-forest-from-scratch
```

### 2. Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

## Random Forest Implementation

Each tree is trained on a bootstrap sample of the training data.

At every node, only a random subset of features is considered.

The best split is selected using reduction in squared error:

```text
Gain =
Parent Error
-
(Left Error + Right Error)
```

The final Random Forest prediction is the average of all individual tree predictions:

```text
Prediction =
(Tree 1 + Tree 2 + ... + Tree N) / N
```

## Dataset

The included dataset contains 600 synthetic student records. It is intended for educational and portfolio purposes.

## Future Improvements

- Add cross-validation
- Add hyperparameter tuning
- Add more student features
- Compare with Gradient Boosting and XGBoost
- Add model performance dashboard
- Deploy the Streamlit application
- Use a real-world education dataset

## Disclaimer

This project is for educational/portfolio purposes. Predictions should not be used to make real educational decisions about students.
