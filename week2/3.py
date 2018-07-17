# coding=utf-8
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.

train_data = pandas.read_csv('perceptron-train.csv', header=None)
test_data = pandas.read_csv('perceptron-test.csv', header=None)

x_train = train_data.loc[:, 1:]
y_train = train_data[0]

x_test = test_data.loc[:, 1:]
y_test = test_data[0]

# Обучите персептрон со стандартными параметрами и random_state=241.

p = Perceptron(random_state=241)
p.fit(x_train, y_train)

# Подсчитайте качество (долю правильно классифицированных объектов, accuracy) 
# полученного классификатора на тестовой выборке.

acc_bfr = accuracy_score(y_test, p.predict(x_test))
print("Accuracy before: %.3f" % acc_bfr)

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
p = Perceptron(random_state=241)
p.fit(x_train_scaled, y_train)
acc_aftr = accuracy_score(y_test, p.predict(x_test_scaled))
print("Accuracy after: %.3f" % acc_aftr)

diff_val = (acc_aftr - acc_bfr)
print(diff_val)