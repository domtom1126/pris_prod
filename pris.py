import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('to_pris.csv')
plea_list = list(df['plea_orcs'].unique())

oe = LabelEncoder()
oe2 = LabelEncoder()
oe3 = LabelEncoder()

df.plea_orcs = oe.fit_transform(df.plea_orcs)
df[['judge']] = oe2.fit_transform(df[['judge']])
df[['race']] = oe3.fit_transform(df[['race']])

X = df[['plea_orcs', 'race', 'prior_cases']]
y = df[['to_prison']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
to_prison = DecisionTreeClassifier()
to_prison.fit(X_train, np.ravel(y_train, order='C'))
print


user_input = [['2913.02', '2', '2']]

prediction = to_prison.predict(user_input)
accuracy = to_prison.score(X_test, y_test)

