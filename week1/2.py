# coding=utf-8
import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

labels = ['Pclass', 'Fare', 'Age', 'Sex']
x = data.loc[:, labels]
x['Sex'] = x['Sex'].map(lambda sex: 1 if sex == 'male' else 0)
y = data['Survived']
x = x.dropna()
y = y[x.index.values]
clf = DecisionTreeClassifier(random_state=241)
clf.fit(np.array(x.values), np.array(y.values))
importances = pandas.Series(clf.feature_importances_, index=labels)
sorted_importances = importances.sort_values(ascending=False).head()
print(sorted_importances)
