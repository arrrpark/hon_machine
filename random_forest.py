import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/car.csv'
data = pd.read_csv(file_url)
print(data.head())
print(round(data.describe(), 2))

data[['engine', 'engine_unit']] = data['engine'].str.split(expand=True)
data['engine'] = data['engine'].astype('float32')
data.drop('engine_unit', axis=1, inplace=True)

data[['max_power', 'max_power_unit']] = data['max_power'].str.split(expand=True)

def isFloat(value):
    try:
        num = float(value)
        return num
    except ValueError:
        return np.nan

data['max_power'] = data['max_power'].apply(isFloat)
data.drop('max_power_unit', axis=1, inplace=True)

data[['mileage', 'mileage_unit']] = data['mileage'].str.split(expand=True)
data['mileage'] = data['mileage'].astype('float32')

def mile(x):
    if x['fuel'] == 'Petrol':
        return x['mileage'] / 80.43
    elif x['fuel'] == 'Diesel':
        return x['mileage'] / 73.56
    elif x['fuel'] == 'LPG':
        return x['mileage'] / 40.85
    else:
        return x['mileage'] / 44.23

data['mileage'] = data.apply(mile, axis=1)
data.drop('mileage_unit', axis=1, inplace=True)

data['torque'] = data['torque'].str.upper()

def torque_unit(x):
    if 'NM' in str(x):
        return 'Nm'
    elif 'KGM' in str(x):
        return 'kgm'

data['torque_unit'] = data['torque'].apply(torque_unit)
# data['torque_unit'].fillna('Nm', inplace=True)

print(data[data['torque_unit'].isna()]['torque'].unique())
data['torque_unit'].fillna('Nm', inplace=True)

def split_num(x):
    x = str(x)
    for i, j in enumerate(x):
        if j not in '0123456789.':
            cut = i
            break
    return x[:cut]

def torque_trans(x):
    if x['torque_unit'] == 'kgm':
        return x['torque'] * 9.8066
    else:
        return x['torque']

data['torque'] = data['torque'].apply(split_num)
data['torque'] = data['torque'].replace('', np.nan)
data['torque'] = data['torque'].astype('float64')
data['torque'] = data.apply(torque_trans, axis=1)
data.drop('torque_unit', axis=1, inplace=True)
data['name'] = data['name'].replace('Land', 'Land Over')

print(data.isna().mean())

data.dropna(inplace=True)
print(len(data))

data = pd.get_dummies(data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

print(data.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('selling_price', axis=1), data['selling_price'], test_size=0.2, random_state=100)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(random_state=100)
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(mean_squared_error(y_train, train_pred) ** 0.5)
print(mean_squared_error(y_test, test_pred) ** 0.5)