# coding=utf-8
import pandas
import re
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.

sex_counts = data['Sex'].value_counts()
print('{} {}'.format(sex_counts['male'], sex_counts['female']))

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

survived_counts = data['Survived'].value_counts()
survived_percent = 100.0 * survived_counts[1] / survived_counts.sum()
print("{0:.2f}".format(survived_percent))

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print("{0:.2f}".format(pclass_percent))

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.

age = data['Age'].dropna() # remove NaN
print("{:0.2f} {:0.2f}".format(age.mean(), age.median()))

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

corr = data['SibSp'].corr(data['Parch'])
print("{:0.2f}".format(corr))


# 6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать
# несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.

f_names = data[data['Sex'] == 'female']['Name']

def extract_first_name(name):
  # Если в скобках, то это имя
  s = re.search(".*\\((.*)\\).*", name)
  if s is not None:
    return s.group(1).split(" ")[0]
  # Удаляем обращения
  s1 = re.search(".*\\. ([A-Za-z]*)", name)
  return s1.group(1)

max_freq = f_names.map(lambda full_name: extract_first_name(full_name)).value_counts().idxmax()
print(max_freq)