import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data_Train1.csv')
dataset["New_Price"].fillna("0", inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 12].values






#categorical encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X0=LabelEncoder()
X[:,0]=labelencoder_X0.fit_transform(X[:,0])

labelencoder_X1=LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])

labelencoder_X4=LabelEncoder()
X[:,4]=labelencoder_X4.fit_transform(X[:,4])


labelencoder_X5=LabelEncoder()
X[:,5]=labelencoder_X5.fit_transform(X[:,5])

labelencoder_X6=LabelEncoder()
X[:,6]=labelencoder_X6.fit_transform(X[:,6])

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='median',axis=0)
imputer=imputer.fit(X[:,:])
X[:,:]=imputer.transform(X[:,:])



from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0,1,4,5,6])
X=onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)



from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=50,criterion='mse',random_state=0)
regressor.fit(X,y)
y_predict=regressor.predict(X)

error=(sum(abs(y-y_predict)))/len(y)

from sklearn.metrics import mean_squared_log_error
rmsle=np.sqrt(mean_squared_log_error( y, y_predict ))
score=1-rmsle


