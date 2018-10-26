import pandas as pd 
import os
from sklearn import preprocessing


x = pd.read_csv('data/train.csv')


# LabelEncoder: str --> num
# OneHotEncoder: num --> onehot


print(x['native_country'].value_counts())
# print(type(x['native_country']))
# print(x)

print("===========PANDAS===========")
native_ohe_1 = pd.get_dummies(x['native_country'])
x_ohe_0 = native_ohe_1.values
print(x_ohe_0)
print(type(native_ohe_1))



print("===========SKLEARN===========")
# OneHotEncoder cant convert string to float
# native_ohe_2 = preprocessing.OneHotEncoder(sparse = False).fit_transform(x['native_country']).reshape((-1, 1))


x_nat = preprocessing.LabelEncoder().fit_transform(x['native_country'])
x_ohe = preprocessing.OneHotEncoder(sparse=False, categories='auto').fit_transform(x_nat.reshape(-1,1))
print(x_ohe)
print(type(x_ohe))

print(x_nat[40:60])




