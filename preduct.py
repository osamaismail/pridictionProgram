import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model



### READ DATA ###

data = pd.read_csv('covid-data.csv', sep=',')
data = data[['id','total_cases']]


### PREPEARE DATA ###
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['total_cases']).reshape(-1,1)
plt.plot(y,'-m')


polyFeat = PolynomialFeatures(degree=9)
x = polyFeat.fit_transform(x)


### TRAINING ###
reg = linear_model.LinearRegression()
reg.fit(x,y)
accuracy = reg.score(x,y)
accu = round(accuracy*100,3)

print(f'Accuracy: {accu} %')
y0 = reg.predict(x)


### PREDUCTION ###
day_future = int(input('Give me the number of days: '))
print(f'Prediction-cases after {day_future} days from {len(data)} is :', end='')
print(round(int(reg.predict(polyFeat.fit_transform([[len(data)+day_future]])))/1000000,3), 'Million Cases')


## Number of days ##
x1 = np.array(list(range(1, len(data)+day_future))).reshape(-1,1)


## Number of Cases  ##
y1 = reg.predict(polyFeat.fit_transform(x1))


## Drowing Graph ##
plt.plot(y1,'--r')
plt.plot(y0,'--b')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()
