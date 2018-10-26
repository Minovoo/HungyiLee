import hw2_logistic_test
import pandas as pd 



train_data_path = 'data/train.csv'
train_label_path = 'data/train.csv'
test_data_path = 'data/test.csv'
output_dir = 'data'
save_dir = 'data'


# X_train, Y_train, X_test = hw2_logistic_test.load_data(train_data_path, train_label_path, test_data_path)

# print(X_train)
# print('\n')
# print(X_test)
# print('\n')
# print(Y_train)




X_train = pd.read_csv(train_data_path)
X_test = pd.read_csv(test_data_path)


print(X_train['native_country'].value_counts())
print(X_test['native_country'].value_counts())




# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)

# print(X_train)
# print(X_test)
# print(Y_train[-4])


# native = X_train[:, -1]

# print(native)


# enc = preprocessing.OneHotEncoder()
# enc.fit(native)
# array = enc.transform

# print '\n'
# print type(X_train[:,-1][0])