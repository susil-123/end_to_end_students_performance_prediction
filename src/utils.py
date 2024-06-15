import os
import pickle
import sys
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            score = r2_score(y_test, model.predict(X_test))
            report[list(models.keys())[i]] = score
        return report

    except Exception as e:
        raise CustomException(e,sys)
