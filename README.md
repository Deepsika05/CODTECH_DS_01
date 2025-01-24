# CODTECH_DS_01

COMPANY   : CODTECH IT SOLUTIONS
NAME      : DEEPSIKA A
INTERN ID : CT08FOT
DOMAIN    : DATA SCIENCE
DURATION  : 4 WEEKS
MENTOR    : NEELA SANTOSH
This task involves creating an exploratory data analysis(EDA) using python and including libraries like pandas,numpy,matplotlib,seaborn which is essential for understanding the dataset, discovering patterns, detecting outliers, and preparing the data for further analysis or machine learning models.

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Titanic Dataset
# If you're using Seaborn, you can load it directly
titanic = sns.load_dataset('titanic')

# Step 3: Basic Data Exploration
print("First 5 rows of the dataset:")
print(titanic.head())  # Display the first 5 rows

# Get a summary of the dataset
print("\nData Info:")
print(titanic.info())

# Get descriptive statistics for numeric columns
print("\nDescriptive Statistics:")
print(titanic.describe())

# Check for missing values
print("\nMissing Values:")
print(titanic.isnull().sum())

# Step 4: Data Cleaning (Handle missing values, if necessary)
# Drop rows where 'age' is missing as a quick approach (can also fill missing values)
titanic_cleaned = titanic.dropna(subset=['age'])

# Step 5: Visualizing the Distribution of Features

# Histograms for numerical features
titanic_cleaned.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Step 6: Visualize Categorical Features

# Count plot for 'Survived' feature
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=titanic_cleaned)
plt.title('Survival Count')
plt.show()

# Count plot for 'Pclass' (passenger class)
plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', data=titanic_cleaned)
plt.title('Passenger Class Distribution')
plt.show()

# Step 7: Explore Relationships Between Variables

# Correlation matrix (for numerical variables)
correlation_matrix = titanic_cleaned.corr()

# Heatmap to visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 8: Scatter Plots and Pair Plots

# Scatter plot between 'Age' and 'Fare'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='fare', data=titanic_cleaned, hue='survived', palette='coolwarm')
plt.title('Age vs Fare (Survived)')
plt.show()

# Pairplot for a subset of features
sns.pairplot(titanic_cleaned[['age', 'fare', 'pclass', 'survived']], hue='survived', palette='coolwarm')
plt.show()

# Step 9: Boxplot for Outliers

# Boxplot for 'Fare' to detect outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='survived', y='fare', data=titanic_cleaned)
plt.title('Fare Distribution by Survival')
plt.show()


