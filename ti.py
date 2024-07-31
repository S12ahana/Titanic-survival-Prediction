import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


titanic_data = pd.read_csv('./Titanic-Dataset.csv')  
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)


X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_preds)}")
print(classification_report(y_train, train_preds))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_train, train_preds), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Training Set')
plt.show()


test_preds = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, test_preds)}")
print(classification_report(y_test, test_preds))


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, test_preds), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Test Set')
plt.show()


