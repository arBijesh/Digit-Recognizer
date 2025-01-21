# Digit Recognizer

## Overview
This project is designed to classify handwritten digits (0-9) using machine learning models. The dataset is sourced from Kaggle's Digit Recognizer competition, and multiple models like Logistic Regression, Random Forest, Gradient Boosting, and LightGBM have been implemented to achieve high accuracy.

---

## Problem Statement
Handwritten digit recognition is a key problem in computer vision and has applications ranging from postal address sorting to bank check processing. The goal of this project is to accurately classify digits using supervised machine learning techniques.

---

## Data Description
The dataset consists of:
- **Train.csv**: Training dataset with pixel values and their corresponding labels (0-9).
- **Test.csv**: Test dataset with pixel values for prediction.
- **Sample_submission.csv**: Sample format for submission of predictions.

All datasets are stored in the `data/` directory.

---

## Key Steps
1. **Data Preprocessing:**
   - Load training and test datasets.
   - Split features and labels for model training.

2. **Modeling:**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - LightGBM

3. **Evaluation:**
   - Accuracy metrics are used to evaluate model performance.
   - Best accuracy achieved: 96.94% with LightGBM.

4. **Submission:**
   - Predictions are saved in the submission files (e.g., `submission_logistic.csv`, `submission_lgbm.csv`).

---

## Results
| Model               | Accuracy  |
|---------------------|-----------|
| Logistic Regression | 91.85%    |
| Decision Tree       | 85.74%    |
| Random Forest       | 96.58%    |
| Gradient Boosting   | 94.00%    |
| LightGBM            | 96.94%    |

---

## Project Structure
