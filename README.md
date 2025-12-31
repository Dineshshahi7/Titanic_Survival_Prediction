# Titanic Survival Prediction using Machine Learning

## Project Overview
This project predicts whether a passenger survived the Titanic shipwreck using machine learning. It uses the classic **Titanic dataset** from Kaggle and demonstrates a full machine learning workflow including data preprocessing, exploratory analysis, feature engineering, and model training. :contentReference[oaicite:1]{index=1}

---

## Problem Statement
Passenger survival on the Titanic depended on multiple factors such as age, gender, class, and family relationships.  
The goal of this project is to build predictive models that can accurately determine **whether a passenger survived or not** based on these attributes.

---

## Dataset Description
The dataset contains passenger information including:
- **PassengerId** – Unique ID for each passenger  
- **Survived** – Target variable (0 = No, 1 = Yes)  
- **Pclass** – Ticket class (1,2,3)  
- **Name**, **Sex**, **Age**  
- **SibSp**, **Parch** – Family relations  
- **Fare**, **Cabin**, **Embarked** – Additional attributes

This dataset is widely used for beginner to intermediate machine learning projects and is sourced from Kaggle’s *Titanic: Machine Learning from Disaster* competition. :contentReference[oaicite:2]{index=2}

---

## Tools & Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Jupyter Notebook  

---

## Data Preprocessing
Before training models, the following steps are performed:
- Handling missing values (e.g., Age, Cabin, Embarked)
- Encoding categorical features (e.g., Sex, Embarked)
- Feature scaling (as needed)
- Feature engineering (e.g., Title extraction, Family size)

These steps help improve model performance and data quality. :contentReference[oaicite:3]{index=3}

---

## Machine Learning Models Used
The following models are typically trained and evaluated:
- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest Classifier**

Among these, **Random Forest** usually achieves strong performance on this dataset. :contentReference[oaicite:4]{index=4}

---

## Model Evaluation
Models are evaluated using metrics like:
- Accuracy  
- Confusion matrix  
- Precision, recall, and F1 score

The goal is to find the best model for survival prediction with reliable performance on unseen data. :contentReference[oaicite:5]{index=5}

---

## Key Insights
- **Women and children** had significantly higher survival rates than men. :contentReference[oaicite:6]{index=6}  
- **Passenger class** influenced survival: upper classes survived more. :contentReference[oaicite:7]{index=7}  
- **Feature engineering** such as extracting titles (Mr, Mrs, etc.) and family size helped improve model performance. :contentReference[oaicite:8]{index=8}  
- Models like Random Forest and Logistic Regression provide solid baselines for prediction tasks. :contentReference[oaicite:9]{index=9}

These insights align with historical and analytical patterns found in the Titanic dataset.

---

## How to Run the Project
This project has been deployed as an interactive web application using Hugging Face Spaces.

You can directly access and test the model here:
🔗 Live Demo: https://huggingface.co/spaces/dineshshahi/titanic-survival-predictor
