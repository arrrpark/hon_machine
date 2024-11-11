import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/salary.csv'
data = pd.read_csv(file_url, skipinitialspace=True)

print(data.head())
print(data.describe())

data['class'] = data['class'].map({ '<=50K' : 0, '>50K': 1})
data.drop('education', axis=1, inplace=True)
print(data['occupation'].value_counts())

country_group = data.groupby('native-country')['class'].mean()
country_group = country_group.reset_index()

data = data.merge(country_group, on='native-country', how='left')
data.drop('native-country', axis=1, inplace=True)
data = data.rename(columns={'class_x': 'class','class_y': 'native-country'})

data['native-country'] = data['native-country'].fillna(-99)
data['workclass'] = data['workclass'].fillna('Private')
data['occupation'] = data['occupation'].fillna('Unknown')

data = pd.get_dummies(data, drop_first=True)
print(data)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.4, random_state=100)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))