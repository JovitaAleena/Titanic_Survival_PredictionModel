import pandas as pd
import numpy as np
import plotext as plt  # ASCII-based plotting for terminal
import matplotlib.pyplot as mpl
import seaborn as sns
from tabulate import tabulate  # Neatly formatted tables
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Titanic dataset
df = pd.read_csv("tested.csv")  # Update path

# Make a copy to avoid modification warnings
df = df.copy()

# Fill missing values correctly
df.loc[:, "Age"] = df["Age"].fillna(df["Age"].mean())
df.loc[:, "Fare"] = df["Fare"].fillna(df["Fare"].mean())
df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert categorical features
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Feature selection
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "SibSp", "Parch"]]
y = df["Survived"]

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to a formatted table
conf_matrix_table = tabulate(conf_matrix, headers=["Predicted: No Survival", "Predicted: Survival"], 
                             tablefmt="grid", showindex=["Actual: No Survival", "Actual: Survival"])

# Print results
print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix_table)

# --- ASCII Visualization - Survival by Gender ---
gender_counts = df.groupby(["Sex", "Survived"]).size().unstack()
plt.bar(["Male Survived", "Male Did Not Survive", "Female Survived", "Female Did Not Survive"], 
        gender_counts.values.flatten())
plt.title("Survival by Gender (ASCII)")
plt.show()

# --- ASCII Visualization - Survival by Passenger Class ---
pclass_counts = df.groupby(["Pclass", "Survived"]).size().unstack()
plt.bar(["1st Class Survived", "1st Class Did Not Survive", 
         "2nd Class Survived", "2nd Class Did Not Survive", 
         "3rd Class Survived", "3rd Class Did Not Survive"], pclass_counts.values.flatten())
plt.title("Survival by Passenger Class (ASCII)")
plt.show()

# --- Matplotlib & Seaborn Graphs ---

# Gender Survival Visualization
sns.countplot(x="Sex", hue="Survived", data=df)
mpl.xticks(ticks=[0, 1], labels=["Male", "Female"])
mpl.title("Survival Count by Gender")
mpl.xlabel("Gender")
mpl.ylabel("Count")
mpl.legend(["Did Not Survive", "Survived"])
mpl.show()

# Passenger Class Survival Visualization
sns.countplot(x="Pclass", hue="Survived", data=df)
mpl.title("Survival Count by Passenger Class")
mpl.xlabel("Passenger Class (1 = First, 3 = Third)")
mpl.ylabel("Count")
mpl.legend(["Did Not Survive", "Survived"])
mpl.show()

# Age Distribution of Survivors
sns.histplot(df[df["Survived"] == 1]["Age"], bins=20, kde=True)
mpl.title("Age Distribution of Survivors")
mpl.xlabel("Age")
mpl.ylabel("Density")
mpl.show()