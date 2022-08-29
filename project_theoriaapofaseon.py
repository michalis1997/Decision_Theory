#imported some libraries
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import validation_curve
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import tensorflow as tf


#we made a function for the Support Vector Machine algorithm
def svm(X_train,X_test ,y_train,  y_test): # takes as arguments X_train,X_test, y_train, y_test

    svm_classifier =  SVC(kernel= 'rbf') #call fuction SVC with kernel Radial Basis Function (rbf) with parameters gamma and C
    svm_classifier.fit(X_train, y_train) # takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred = svm_classifier.predict(X_test) #enables to predict the labels of the data on the basis of the trained model

    cm_test = confusion_matrix(y_pred, y_test) # measure the performance of Support Vector Machine algorithm

    y_pred_train = svm_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    #print metrics for SVM
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
     #print the accuracy of training set for SVM and the accuracy test set
    print('\nAccuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    title = "Confusion matrix, without normalization for SVM" #define the title of confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator( #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
            svm_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues #we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)

    print(title)#display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show() #show the confusion matrix

    return svm_classifier #we return svm_classifier

#we made a function for the Naive Bayes algorithm
def Naive_Bayes(X,y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    nb_classifier = GaussianNB() #classify the multivariate Gaussian model
    nb_classifier.fit(X_train, y_train)# takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred =nb_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = nb_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    title = "Confusion matrix, without normalization for Naive Bayes" #define the title of confusion matrix
    #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
            nb_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues #we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)

    print(title) #display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    #print metrics for Naive Bayes
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for Naive Bayes and the accuracy test set
    print('\nAccuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return nb_classifier #we return nb_classifier

#we made a function for the Logistic Regression algorithm
def Logistic_Regresion(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lr_classifier = LogisticRegression(max_iter=1000) #classify with maximum iterations 1000 for the solvers to converge
    lr_classifier.fit(X_train, y_train) #takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred = lr_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = lr_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Logistic Regression" #define the title of confusion matrix
    #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
            lr_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues  #we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)
    print(title) #display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    #print metrics for Logistic Regression
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for Logistic Regression and the accuracy test set
    print('\nAccuracy for training set for Logistic Regression = {}'.format(
        (cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return lr_classifier #we return lr_classifier

#we made a function for the Decision Tree algorithm
def Decision_Tree(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    dt_classifier = DecisionTreeClassifier()#measure the quality of a split
    dt_classifier.fit(X_train, y_train)#takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred = dt_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = dt_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Decision Tree"#define the title of confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(  #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
            dt_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues,#we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)
    print(title) #display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    #print metrics for Decision Tree
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for Decision Tree and the accuracy test set
    print('\nAccuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return dt_classifier #we return dt_classifier

#we made a function for the Random Forest algorithm
def Random_Forest(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    rf_classifier = RandomForestClassifier(n_estimators=10)#meta estimator that fits a number of decision tree classifiers.
    # We have 10 estimators=> we have 10 trees that we want to build before taking the maximum voting or averages of predictions
    rf_classifier.fit(X_train, y_train)#takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred = rf_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = rf_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Random Forest" #define the title of confusion matrix
    #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
            rf_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues#we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)

    print(title)#display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    #print metrics for Random Forest
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for Random Forest and the accuracy test set
    print('\nAccuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(random_state=1)
    # define search space
    space = dict()
    space['n_estimators'] = [10, 100, 500]
    space['max_features'] = [2, 4, 6]
    # define search
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # execute the nested cross-validation
    scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    return rf_classifier #renturn rf_classifier

#we made a function for the XGBoost algorithm

def XGBoost(X_train,X_test, y_train,y_test):# takes as arguments X_train,X_test, y_train, y_test

    xg = XGBClassifier(use_label_encoder=False) #remove user warning that comes while running XGBClassifier.
    xg.fit(X_train, y_train)#takes the training data as arguments to achieve better accuracy

    # Predicting the Test set results
    y_pred = xg.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = xg.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:  # setting threshold to .5
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for XGBoost"#define the title of confusion matrix
    #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
            xg,
            X_test,
            y_test,
            cmap=plt.cm.Blues#we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)
    print(title)#display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    #print metrics for XGBoost
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for XGBoost  and the accuracy test set
    print('\nAccuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return xg #return xg

#we made a function for the KNN algorithm
def Kneighbors(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    knn_classifier = KNeighborsClassifier(n_neighbors = 10)# tell the classifier to use Euclidean distance for determining the proximity between neighboring points
    knn_classifier.fit(X_train,y_train)#takes the training data as arguments to achieve better accuracy


    # Predicting the Test set results
    y_pred = knn_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = knn_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for KNN"#define the title of confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator( #we call ConfusionMatrixDisplay function with some arguments to create the confusion matrix
            knn_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues,#we set color blue for the confusion matrix
        )
    disp.ax_.set_title(title)
    print(title)#display the title of confusion matrix
    print(disp.confusion_matrix)
    plt.show()#show the confusion matrix

    # Setting the range for the parameter (from 1 to 10)
    parameter_range = np.arange(1, 10, 1)

    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                               param_name = "n_neighbors",
                                               param_range = parameter_range,
                                               cv = 5, scoring = "accuracy")
    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
         label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score,
         label = "Cross Validation Score", color = 'g')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

    #print metrics for KNN
    print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
    #print the accuracy of training set for KNN  and the accuracy test set
    print('\nAccuracy for training set for Kneighbors = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Kneighbors = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return knn_classifier #return knn_classifier


def deep_learning(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train = tf.keras.utils.normalize(X_train, axis =1)
    X_test = tf.keras.utils.normalize(X_test, axis =1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimazer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,y_train, epochs=3)
    val_loss,val_acc = model.evaculate(X_test,y_test)

    print(val_loss,val_acc)

def main():

    df = pd.read_csv("Dataset 1.csv", delimiter=",") #read the dataset
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target'] #we define the labales of each column
    print(df.isnull().sum())#returns the number of missing values in the dataset.


    df['sex'] = df.sex.map({0: 'female', 1: 'male'})#we map 0 as female and 1 as male


    #plots
    sns.set_context("paper", font_scale = 1, rc = {"font.size": 18,"axes.titlesize": 20,"axes.labelsize": 20})
    sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
    plt.title('Variation of Age for each target class')
    plt.show()

    # barplot of age vs sex with hue = target
    sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
    plt.title('Distribution of age vs sex with the target class')
    plt.show()

    df['sex'] = df.sex.map({'female': 0, 'male': 1})

    # data preprocessing
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    sc = ss()# Create a state space system.
    X_train = sc.fit_transform(X_train) # training data so that we can scale the training data and also learn the scaling parameters of that data.
    X_test = sc.transform(X_test) # transforming all the features using the respective mean and variance.

    #  SVM
    svm_classifier = svm(X_train, X_test, y_train,  y_test)


    #  Naive Bayes
    nv_classifier = Naive_Bayes(X, Y)

    #  Logistic Regression
    lr_classifier = Logistic_Regresion(X, Y)

    #  Decision Tree
    dt_classifier = Decision_Tree(X, Y)

    # Random Forest
    rf_classifier = Random_Forest(X,Y)

    # applying XGBoost
    xgb_classifier = XGBoost(X_train, X_test, y_train, y_test)

    # Kneighbors Classifier
    kn_classifier = Kneighbors(X, Y)

    deep_learning(X,Y)

    test = [[65, 1, 0, 200, 250, 1, 1, 180, 1, 2.5, 2, 3, 3],
            [100, 1, 0, 200, 300, 1, 2, 200, 1, 2.5, 2, 3, 3],
            [52, 1, 0, 200, 150, 1, 0, 150, 1, 2.3, 0, 1, 3],
            [45, 0, 3, 130, 200, 1, 2, 180,0 ,2.1, 2, 1 ,3],
            [55, 1, 3, 180, 140, 0, 2, 180,0 ,2.5, 2, 3 ,3]]

    test = sc.fit_transform(test)
    # Test models for 5 cases

    for i in range(len(test)): # print target for each algorithm that tell us if a patient suffers from heart disease
        print(f"Patient {i} is: \n")
        print(f"\t SVM: {svm_classifier.predict([test[i]])}" )
        print(f"\t Naive Bayes: {nv_classifier.predict([test[i]])}")
        print(f"\t Logistic Regression: {lr_classifier.predict([test[i]])}")
        print(f"\t Decision Tree: {dt_classifier.predict([test[i]])}")
        print(f"\t Random Forest: {rf_classifier.predict([test[i]])}")
        print(f"\t XGBoost: {xgb_classifier.predict([test[i]])}")
        print(f"\t KNeighbors: {kn_classifier.predict([test[i]])}")
        print()

if __name__ == "__main__":
    main()
