import pandas as pd
import numpy as np
import dill
import os
import sys
from sklearn.metrics import r2_score
from src.exception import  CustomeException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred_test=model.predict(X_test)
            test_model_score=r2_score(y_test,y_pred_test)
            report[list(models.keys())[i]]=test_model_score
            return report
        
    except Exception as e:
        raise CustomeException(e,sys)
        