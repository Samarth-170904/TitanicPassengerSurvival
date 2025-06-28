import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# from sklearn import set_config
# set_config(display='diagram')

df = pd.read_csv("Titanic.csv")
# print(df.head())

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

x_tain,x_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)
# print(x_tain.head())

trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embark',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

trf2 = ColumnTransformer([
    ('ohe_sex_emb',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')

trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])

trf4 = DecisionTreeClassifier(max_depth=5)

pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4)
])

# pipe.fit(x_tain,y_train)

# y_pred = pipe.predict(x_test)
# accuracy = accuracy_score(y_test,y_pred)
# print(f"Accuracy before cross validation = {accuracy*100} %")

# accuracy2 = cross_val_score(pipe,x_tain,y_train,cv=5,scoring='accuracy').mean()
# print(f"Accuracy after cross validation = {accuracy2*100} %")

param_grid = {
    'trf4__max_depth':[3,5,10,None],
    'trf4__min_samples_split':[2,5,10],
    'trf4__criterion':['gini','entropy']
}

search = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
search.fit(x_tain,y_train)

print(f"Best Accuracy = {search.best_score_}")
print(f"Best Parameter = {search.best_params_}")