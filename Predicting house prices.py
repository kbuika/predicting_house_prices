import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
# import the ipython function if the %matplotlib inline does not work.
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
	get_ipython().run_line_magic('matplotlib', 'inline')


sacra = pd.read_csv(r"C:\Users\Steve\Desktop\ML/Sacramentorealestatetransactions.csv")


sacra.head()


sacra.columns


# This works to remove the columns with strings in our dataset.
columns = ['street', 'city','state', 'type']
sacra.drop(columns, inplace=True, axis=1)
sacra


sacra.head()


sacra.describe()

sacra['beds'].value_counts().plot(kind='bar')
plt.title('Number of Beds')
plt.xlabel('Beds')
plt.ylabel('Count')
sns.despine


# # Visualizing the location of the houses.

# According to the dataset, we have latitude and longitude on the dataset
# for each house. we are going to see the common location.
# its pretty awesome
plt.figure(figsize=(10,10))
sns.jointplot(x=sacra.latitude.values, y=sacra.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.grid()
plt.show()
sns.despine


## factors affecting the house prices
a = sacra.loc[sacra['sq__ft']==0]
a.index
b = a.index
b
z = sacra.drop(b)

t = sacra.loc[sacra['sq__ft']>5000]
t
z = z.drop(866)
z

plt.scatter(z.sq__ft,z.price)
plt.title("Price vs Square Feet")
plt.xlabel("Sq_Feet")
plt.ylabel("Price")
plt.show()




plt.scatter(sacra.price,sacra.longitude)
plt.title("Price vs Longitude")
plt.show()

# # Linear Regression

# Is a model in statistics which enables us predict the future based upon
# past relationship of variables.
# y = mx + c


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

labels = z['price']
conv_dates = [1 if values == 2008 else 0 for values in z.sale_date]
z['sale_date'] = conv_dates
train1 = z.drop(['price'], axis=1)

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train , y_test = train_test_split(train1, labels,test_size = 0.01, random_state = 2)

reg.fit(x_train,y_train)

reg.score(x_test,y_test)


# So our accuracy is just 20%... that is low
# In order to improve our prediction, we can use Gradient Boosting Regression.
# It is a machine learning technique for regression and classification problems,
#which produces a prediction model in the form of and ensemble of weak prediction models, typically decision trees.

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 10, min_samples_split = 2,
                                        learning_rate = 0.01, loss = 'ls')


clf.fit(x_train, y_train)

clf.score(x_test,y_test)

from sklearn.grid_search import RandomizedSearchCV

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

