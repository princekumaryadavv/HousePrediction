# Load necessary libraries
import numpy as np
import pandas as pd
from category_encoders import MEstimateEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Load the dataset
df = pd.read_csv(r"C:\Users\princ\Downloads\data.csv")

# Define a function to calculate mutual information scores
def make_mi_score(X, y):
    X_copy = X.copy()
    for colname in X.select_dtypes("object"):
        X_copy[colname], _ = X_copy[colname].factorize()

    discrete_features = X_copy.dtypes == int
    mi_scores = mutual_info_regression(X_copy, y, discrete_features=discrete_features, random_state=42)
    mi_scores = pd.Series(mi_scores, name='Mutual Information', index=X_copy.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

# Define a function to plot mutual information scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

# Select relevant columns and target variable
df = df[['street', 'statezip', 'city', 'sqft_living', 'sqft_above', 'bathrooms', 'sqft_lot', 'price']]

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df[['price']]

# Encode categorical features
encoder = MEstimateEncoder(cols=['street', 'statezip', 'city'], m=0.5)
X_encoded = encoder.fit_transform(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.4, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict house price based on user input
def predict_house_price():
    features = {}
    for column in X.columns:
        value = input(f"Enter {column}: ")
        features[column] = [value]
    features_df = pd.DataFrame(features)
    features_encoded = encoder.transform(features_df)
    prediction = model.predict(features_encoded)
    print(f"Predicted Price: ${prediction[0][0]}")

# Interactive prediction
while True:
    predict_house_price()
    cont = input("Do you want to predict another house price? (y/n): ")
    if cont.lower() != 'y':
        break
