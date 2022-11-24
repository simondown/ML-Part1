import pandas as pd
import numpy as np

np.random.seed(42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

le = LabelEncoder()

df = pd.read_csv('TRAIN.csv')

df.drop(df.columns[0], axis = 1, inplace = True)
print(df.columns)
#print(df.count(axis = 0))
#print(df.shape)
#print(df.describe())
#print(df.dtypes)
df['cut'] = le.fit_transform(df['cut'])
df['color'] = le.fit_transform(df['color'])
df['clarity'] = le.fit_transform(df['clarity'])
#print(df)

X = np.array(df.drop(columns = 'price'))
Y = np.array(df['price'])

X, Y = shuffle(X, Y, random_state = 42)

scores_final = []

print(X)
print(Y)

reg1 = DecisionTreeRegressor(criterion = 'squared_error', max_depth = 12, random_state = 42)
scores = cross_val_score(reg1, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

reg2 = DecisionTreeRegressor(criterion = 'friedman_mse', max_depth = 16, random_state = 42)
scores = cross_val_score(reg2, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

reg3 = DecisionTreeRegressor(criterion = 'poisson', max_depth = 22, random_state = 42)
scores = cross_val_score(reg3, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

reg4 = DecisionTreeRegressor(criterion = 'squared_error', max_depth = 45, random_state = 42)
scores = cross_val_score(reg4, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

reg5 = DecisionTreeRegressor(criterion = 'friedman_mse', max_depth = 95, random_state = 42)
scores = cross_val_score(reg5, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

reg6 = DecisionTreeRegressor(criterion = 'poisson', max_depth = 3, random_state = 42)
scores = cross_val_score(reg6, X, Y, cv=10, scoring = 'r2')
scores_final.append(scores.mean())

print(scores_final)





