import pandas as pd
import numpy as np
import logging

import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

def modelisation(X_train: pd.DataFrame , y_train: pd.DataFrame , X_test: pd.DataFrame ,y_test: pd.DataFrame  ):

    preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))
    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    model = RandomForest
    
    model.fit(X_train, y_train)

    model_save = pickle.dumps(model)

    return model_save


def prediction(model_save , X_test: pd.DataFrame , y_test: pd.DataFrame ) :
    model = pickle.loads(model_save) 
    ypred_nparray = model.predict(X_test)

    ypred = pd.DataFrame(ypred_nparray)


    #print("CONFUSION MATRIX",confusion_matrix(y_test, ypred))
    report=classification_report(y_test, ypred,output_dict=True)
    report1=classification_report(y_test, ypred)
    print(report1)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    print(df_classification_report)
    score = accuracy_score(y_test,ypred)

    return df_classification_report

    