import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns
import time
%matplotlib inline 

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn import metrics




data = pd.read_csv("data.csv")

data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# print(data.groupby('diagnosis').size())
# sns.countplot(data['diagnosis'],label="Count")
# plt.show()



# features_mean= list(data.columns[1:11])
# corr = data[features_mean].corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
#            xticklabels= features_mean, yticklabels= features_mean,
#            cmap= 'coolwarm') 
# plt.show()


Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=21)

inp = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
out ='diagnosis'


mode = LogisticRegression()
# mode = DecisionTreeClassifier()
# mode = KNeighborsClassifier()
# mode = SVC()


mode.fit(data[inp],data[out])
  
  #Make predictions on training set:
predictions = mode.predict(data[inp])
  
  #Print accuracy
accuracy = metrics.accuracy_score(predictions,data[out])
print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
kfold = KFold(data.shape[0], n_folds=5)
error = []
for train, test in kfold:
    # Filter training data
    train_predictors = (data[inp].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))





Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=21)
models_list = []
models_list.append(('LR', LogisticRegression()))
models_list.append(('DT', DecisionTreeClassifier()))
models_list.append(('SVM', SVC())) 
models_list.append(('KNN', KNeighborsClassifier()))
num_folds = 10
results = []
names = []
for name, model in models_list:
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=num_folds, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s:(run time: %f)" % (name, end-start))