#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Using--Linear--,-Lasso-,-Ridge--Regression" data-toc-modified-id="Using--Linear--,-Lasso-,-Ridge--Regression-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Using  Linear  , Lasso , Ridge  Regression</a></span></li><li><span><a href="#Using-Polynomial-Regression" data-toc-modified-id="Using-Polynomial-Regression-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Using Polynomial Regression</a></span></li><li><span><a href="#Using-Random-Forest-Regression" data-toc-modified-id="Using-Random-Forest-Regression-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Using Random Forest Regression</a></span></li></ul></div>

# In[1]:


#NumPy is a python library used for working with arrays.
#Also it is optimized to work with latest CPU architectures.

#pandas is a Python package providing fast, flexible, and 
#expressive data structures designed to make working with
#structured and time series data both easy and intuitive. 

import numpy as np  #linear algebra 
import pandas as pd  #data processing (Eg: df.read_csv)


# In[2]:


#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('Concrete_Dataset.csv')
print(df.shape)
df.head()


# In[4]:


#co-relation matrix
df_1 = df[['cement','slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age', 'csMPa']]
df_1.corr()


# In[5]:


corr = df_1.corr()
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.heatmap(corr,xticklabels=True,yticklabels=True,annot = True,cmap ='RdBu_r')
plt.title("Correlation Between Variables")
plt.tight_layout()
plt.savefig('Correlation_Between_Variables.png')


# In[6]:


# pair Plot
sns.pairplot(df_1,palette="husl",diag_kind="kde")
plt.savefig('pair_plot.png')


# ## Using  Linear  , Lasso , Ridge  Regression

# In[7]:


from sklearn.linear_model import LinearRegression , Lasso , Ridge
lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()


# In[8]:


X = df[['cement','slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']].values
Y = df[['csMPa']].values
        


# In[9]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3 , random_state = 2)


# In[10]:


lr.fit(xtrain, ytrain)
lasso.fit(xtrain, ytrain)
ridge.fit(xtrain, ytrain)


# In[11]:


y_p_linear = lr.predict(xtrain)
y_p_lasso = lasso.predict(xtrain)
y_p_ridge = ridge.predict(xtrain)


# In[12]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#training accuracy 

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_linear)),mean_squared_error(ytrain, y_p_linear),
            mean_absolute_error(ytrain, y_p_linear), r2_score(ytrain, y_p_linear)))
print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_lasso)),mean_squared_error(ytrain, y_p_lasso),
            mean_absolute_error(ytrain, y_p_lasso), r2_score(ytrain, y_p_lasso)))
print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_ridge)),mean_squared_error(ytrain, y_p_ridge),
            mean_absolute_error(ytrain, y_p_ridge), r2_score(ytrain, y_p_ridge)))


# In[13]:


#testing accuracy

lr.fit(xtest, ytest)
lasso.fit(xtest, ytest)
ridge.fit(xtest, ytest)

yp_test_linear = lr.predict(xtest)
yp_test_lasso = lasso.predict(xtest)
yp_test_ridge = ridge.predict(xtest)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_linear)),mean_squared_error(ytest, yp_test_linear),
            mean_absolute_error(ytest, yp_test_linear), r2_score(ytest, yp_test_linear)))
print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_lasso)),mean_squared_error(ytest, yp_test_lasso),
            mean_absolute_error(ytest, yp_test_lasso), r2_score(ytest, yp_test_lasso)))
print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_ridge)),mean_squared_error(ytest, yp_test_ridge),
            mean_absolute_error(ytest, yp_test_ridge), r2_score(ytest, yp_test_ridge)))


# In[14]:


print(ytest.shape)
print(xtest.shape)
print(yp_test_linear.shape)
print(y_p_linear.shape)


# In[15]:


#Prediction Plot of cement using Linear Regression.

plt.plot(figsize=(12,4))
plt.title('Linear Regression Cement - Compressive Strength Vs Cement')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(xtest[:, 1] , ytest , c='r')
plt.scatter(xtest[: , 1] , yp_test_linear)
plt.show()


# In[16]:


#Prediction Plot of cement using Lasso Regression.

plt.plot(figsize=(12,4))
plt.title('Lasso Regression Cement - Compressive Strength Vs Cement')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(xtest[:, 1] , ytest , c='r')
plt.scatter(xtest[: , 1] , yp_test_lasso)
plt.show()


# In[17]:


#Prediction Plot of cement using Ridge Regression.

plt.plot(figsize=(12,4))
plt.title('Ridge Regression Cement - Compressive Strength Vs Cement')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(xtest[:, 1] , ytest , c='r')
plt.scatter(xtest[: , 1] , yp_test_ridge)
plt.show()


# In[18]:


plt.plot(figsize=(12,4))
plt.title('Linear Regression')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(yp_test_linear , ytest , c='r')
plt.plot([ytest.min() , ytest.max()] , [ytest.min() , ytest.max()])

plt.savefig('True_Vs_Predicted_linear.png')
plt.show()


# In[19]:


plt.plot(figsize=(12,4))
plt.title('Lasso Regression')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(yp_test_lasso , ytest , c='r')
plt.plot([ytest.min() , ytest.max()] , [ytest.min() , ytest.max()])

plt.savefig('True_Vs_Predicted_lasso.png')
plt.show()


# In[20]:


plt.plot(figsize=(12,4))
plt.title('Ridge Regression')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(yp_test_ridge , ytest , c='r')
plt.plot([ytest.min() , ytest.max()] , [ytest.min() , ytest.max()])

plt.savefig('True_Vs_Predicted_ridge.png')
plt.show()


# ## Using Polynomial Regression

# In[21]:


X = df[['cement','slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']].values
Y = df[['csMPa']].values


# In[22]:


import numpy as np
import sklearn 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3 , random_state = 2)

from sklearn.linear_model import LinearRegression
from sklearn import model_selection


# Here we're using **Cross Validation - Kfold** just to make sure which degree we should prefer the most.

# In[23]:


scores = []
for i in range(1, 10):
    pol = PolynomialFeatures(degree = i)
    x_pol = pol.fit_transform(xtrain)

    kfold = model_selection.KFold(n_splits=8)
    model_kfold = LinearRegression()
    result_kfold = model_selection.cross_val_score(model_kfold , x_pol , ytrain , cv=kfold)

    scores.append(result_kfold.mean()*100)


# In[24]:


scores


# In[25]:


plt.plot(range(1, 10) , scores)
plt.yscale('log')
plt.show()


# In[26]:


from sklearn import model_selection
pol = PolynomialFeatures(degree = 3)
x_pol = pol.fit_transform(xtrain)

kfold = model_selection.KFold(n_splits=8)
model_kfold = LinearRegression()
result_kfold = model_selection.cross_val_score(model_kfold , x_pol , ytrain , cv=kfold)

print('Accuracy:' , (result_kfold.mean()*100))


# In[27]:


lmodel = LinearRegression() 
lmodel.fit(x_pol,ytrain)


# In[28]:


ypp = lmodel.predict(x_pol)
ypp


# In[29]:


print(xtest.shape)
print(ytest.shape)


# In[30]:


print(x_pol.shape) #we have 8 independent features.
print(ypp.shape)


# In[31]:


print(Y.shape)
print(X.shape)


# In[32]:


#Prediction Plot of cement using Polynomial Regression.

plt.plot(figsize=(12,4))
plt.title('Polynomial Regression Cement - Compressive Strength Vs Cement')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(xtrain[:, 1] , ytrain)
plt.scatter(xtrain[:,0:1], ypp,c='r')
plt.show()


# In[33]:


#training accuracy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""PolynomialRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, ypp)),mean_squared_error(ytrain, ypp),
            mean_absolute_error(ytrain, ypp), r2_score(ytrain, ypp)))


# In[34]:


#testing accuracy

x_test_pol = pol.fit_transform(xtest)
yp_t = lmodel.predict(x_test_pol)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""PolynomialRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_t)),mean_squared_error(ytest, yp_t),
            mean_absolute_error(ytest, yp_t), r2_score(ytest, yp_t)))


# In[35]:


#Final plotting using Polynomial Regression.

plt.plot(figsize=(12,4))
plt.title('Polynomial Regression')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(ypp , ytrain , c='y')
plt.plot([ytest.min() , ytest.max()] , [ytest.min() , ytest.max()], c='r')

plt.savefig('True_Vs_Predicted_polynomial.png')
plt.show()


# ## Using Random Forest Regression

# In[36]:


X = df[['cement','slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']].values
Y = df[['csMPa']].values


# In[37]:


from sklearn.ensemble import RandomForestRegressor
rmodel = RandomForestRegressor()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y ,test_size = 0.3 , random_state = 2)

from sklearn.metrics import r2_score


# In[38]:


rmodel.fit(xtrain, ytrain)


# In[39]:


y_p_tree_train = rmodel.predict(xtrain)
y_p_tree_test = rmodel.predict(xtest)


# In[40]:


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
print(y_p_tree_train.shape)
print(y_p_tree_test.shape)


# In[41]:


#Prediction Plot of cement using Random Forest Regression.

plt.plot(figsize = (12, 4))
plt.title('Random Forest Tree - Compressive Strength Vs Cement') 
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(xtrain[:, 1] , ytrain)
plt.scatter(xtrain[:, 1] , y_p_tree_train , c='r')
plt.show()


# In[42]:


#training accuracy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""Random Forest Tree \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_tree_train)),mean_squared_error(ytrain, y_p_tree_train),
            mean_absolute_error(ytrain, y_p_tree_train), r2_score(ytrain, y_p_tree_train)))


# In[43]:


#testing accuracy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""Random Forest Tree \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, y_p_tree_test)),mean_squared_error(ytest, y_p_tree_test),
            mean_absolute_error(ytest, y_p_tree_test), r2_score(ytest, y_p_tree_test)))


# In[44]:


#Final Plotting

plt.plot(figsize = (12, 4))
plt.title('Random Forest Tree Regression')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.scatter(y_p_tree_train , ytrain , c='y')
plt.plot([ytest.min() , ytest.max()] , [ytest.min() , ytest.max()], c='r')

plt.savefig('True_Vs_Predicted_Random_Forest_Tree.png')
plt.show()


# In[45]:


labels = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Polynomial Regressor", "Random Forest Tree Regressor"]
training_accuracy = [0.62 , 0.62 , 0.62 , 0.93 , 0.98]
testing_accuracy = [0.61 ,  0.61 , 0.61 , 0.80 , 0.91]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig,ax = plt.subplots(figsize = (8 , 8))
rects1 = ax.bar(labels, training_accuracy, label='training')
rects2 = ax.bar(labels , testing_accuracy, label='testing')


# Text and title for labels.
ax.set_ylabel('R2_score')
ax.set_title('R2_score of different Algorithms')
ax.set_xticks(x)
ax.set_xticklabels((labels) , rotation = 30)
ax.legend()

fig.tight_layout()
plt.savefig('R2_score_of_different_Algorithms.png')

plt.show()


# In[46]:


training_accuracy = [0.62 , 0.62 , 0.62 , 0.93 , 0.98]
testing_accuracy = [0.61 ,  0.61 , 0.61 , 0.80 , 0.91]

plt.title('Accuracy')

plt.plot( training_accuracy, label='training accuracy')
plt.plot(testing_accuracy, label='testing accuracy' )

plt.legend()
fig.tight_layout()
plt.savefig('Accuracy.png')
plt.show()


# In[47]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#training accuracy 
print('TRAINING ACCURACY')

print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_linear)),mean_squared_error(ytrain, y_p_linear),
            mean_absolute_error(ytrain, y_p_linear), r2_score(ytrain, y_p_linear)))
print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_lasso)),mean_squared_error(ytrain, y_p_lasso),
            mean_absolute_error(ytrain, y_p_lasso), r2_score(ytrain, y_p_lasso)))
print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_ridge)),mean_squared_error(ytrain, y_p_ridge),
            mean_absolute_error(ytrain, y_p_ridge), r2_score(ytrain, y_p_ridge)))


print("""PolynomialRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, ypp)),mean_squared_error(ytrain, ypp),
            mean_absolute_error(ytrain, ypp), r2_score(ytrain, ypp)))


print("""Random Forest Tree \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytrain, y_p_tree_train)),mean_squared_error(ytrain, y_p_tree_train),
            mean_absolute_error(ytrain, y_p_tree_train), r2_score(ytrain, y_p_tree_train)))


# In[48]:


print('TESTING ACCURACY')
print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2_score")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_linear)),mean_squared_error(ytest, yp_test_linear),
            mean_absolute_error(ytest, yp_test_linear), r2_score(ytest, yp_test_linear)))
print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_lasso)),mean_squared_error(ytest, yp_test_lasso),
            mean_absolute_error(ytest, yp_test_lasso), r2_score(ytest, yp_test_lasso)))
print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_test_ridge)),mean_squared_error(ytest, yp_test_ridge),
            mean_absolute_error(ytest, yp_test_ridge), r2_score(ytest, yp_test_ridge)))


print("""PolynomialRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, yp_t)),mean_squared_error(ytest, yp_t),
            mean_absolute_error(ytest, yp_t), r2_score(ytest, yp_t)))


print("""Random Forest Tree \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(ytest, y_p_tree_test)),mean_squared_error(ytest, y_p_tree_test),
            mean_absolute_error(ytest, y_p_tree_test), r2_score(ytest, y_p_tree_test)))


# In[ ]:




