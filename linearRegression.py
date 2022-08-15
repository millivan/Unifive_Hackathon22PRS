import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
from matplotlib import style, pyplot


df = pd.DataFrame(pd.read_excel("MASA_Hackathon_2022_Travel_Insurance_Data_Set_Cleaned.xlsx"))

data = df[["Product Name", "Destination", "Duration", "Net Sales"]]

data = pd.get_dummies(data, columns=["Product Name", "Destination"])

predict = "Net Sales"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# TRANING PROCESS
# best = 0
# count = 1
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     # Using Ridge Regression model
#     linear = linear_model.Ridge()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(count, ":", acc)
#
#     if acc > best:
#         with open("unifive_netsales_boosting_model_v1.pickle", "wb") as file:
#             pickle.dump(linear, file)
#         best = acc
#     count += 1
# print("Best accurary: ", best)

# Load model
pickle_in = open("unifive_netsales_boosting_model_v1.pickle", "rb")
linear = pickle.load(pickle_in)

# Plot graph
p = "Duration"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Net Sales")
pyplot.show()

