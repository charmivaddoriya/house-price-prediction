# House Price Prediction

A machine learning project that predicts median house values using property and demographic features from the California Housing dataset.

The project demonstrates a complete machine learning workflow, including data acquisition, exploratory data analysis, preprocessing, model training, evaluation, prediction, and automated testing.

---

## Problem Statement

House prices depend on multiple factors such as income, property characteristics, population, and geographic location.

The goal of this project is to build a Linear Regression model that predicts median house values based on available housing and demographic features.

---

## Project Objectives

- Load and inspect a real-world housing dataset
- Perform exploratory data analysis
- Analyze feature relationships and correlations
- Prepare data for machine learning
- Train a Linear Regression model
- Evaluate model performance using regression metrics
- Predict house values for new property data
- Create a reusable Python-based ML pipeline
- Add automated tests for data and model functionality

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Exploratory analysis |
| Pytest | Automated testing |
| Joblib | Model serialization |

---

## Dataset

This project uses the California Housing dataset provided through Scikit-learn.

The dataset contains:

- **20,640 observations**
- **8 input features**
- **1 target variable**

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

### Target

`MedHouseVal` represents the median house value in units of $100,000.

---

## Machine Learning Workflow

```text
California Housing Dataset
          |
          v
     Data Loading
          |
          v
 Exploratory Data Analysis
          |
          v
 Feature / Target Selection
          |
          v
     Train/Test Split
          |
          v
   Linear Regression
          |
          v
      Predictions
          |
          v
 Model Evaluation
          |
          v
    Saved ML Model
          |
          v
   New House Prediction