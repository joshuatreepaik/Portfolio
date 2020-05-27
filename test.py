import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #able to graph data
import pickle #used to save model
from matplotlib import style

#reads the dataset
data = pd.read_csv("student-por.csv", sep=";")

#Get only G1,G2,G3,studytime,failures,absences attributes
data = data[["G1","G2","G3","studytime","failures", "absences"]]

#Going to print this attribute
predict = "G3"

#array without G3
x = np.array(data.drop([predict],1))
#array with every attributes
y = np.array(data[predict])

x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
'''
#save better model
best = 0
for _ in range(30):
	#Get some portion of x and y. split 90% of the data to the training set while 10% of the data to test set using below code.
	x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

	linear = linear_model.LinearRegression()
	#train the algorithm
	linear.fit(x_train, y_train)
	#find the accuracy
	acc = linear.score(x_test, y_test)
	print(acc)
	#only save model if acc is higher than better 
	if acc>best:
		best = acc
		#save pickle file
		with open("studentmodel.pickle", "wb") as f:
			pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle","rb")

#load the pickle into our linear model
linear = pickle.load(pickle_in)

#higher coefficient have higher weight when predicting G3
print('Coefficient: \n', linear.coef_) 
print('Intercept: \n', linear.intercept_)


predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print(predictions[x],x_test[x], y_test[x])

p = 'absences'
style.use("ggplot")
#make scatterplot
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

