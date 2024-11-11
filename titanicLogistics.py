import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/titanic.csv'
data = pd.read_csv(file_url)
data = data.drop(['Name', 'Ticket'], axis=1)

print(data.head())
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', vmin=-1, vmax=1, annot=True)
# plt.show()
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

from sklearn.model_selection import train_test_split
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

