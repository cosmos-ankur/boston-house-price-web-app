import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.ensemble import GradientBoostingRegressor
grad = GradientBoostingRegressor()
grad.fit(x_train,y_train)
print(grad.score(x_test,y_test))
import pickle
# open a file, where you ant to store the data
file = open('boston_model.pkl', 'wb')

# dump information to that file
pickle.dump(grad, file)
print(x_train)