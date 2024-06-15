import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainingConfig:
    model_training_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.ModelTrainingConfig = ModelTrainingConfig()
    
    def initiate_model_training(self,train,test):
        try:

            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }
        
            X_train,y_train,X_test,y_test = (
                train[:,:-1],
                train[:,-1],
                test[:,:-1],
                test[:,-1]
            )

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)

            max_model_score = max(list(model_report.values()))
            max_model_name  = list(models.keys())[list(model_report.values()).index(max_model_score)]
            best_model = models[max_model_name]

            if max_model_score < 0.6:
                raise CustomException("No models were able to acheive the minimum of 60 percent accuracy",sys)
            logging.info(f"Best found model on both training and testing dataset: {max_model_name} with the r2_score of: {max_model_score}")

            
            save_obj(file_path=self.ModelTrainingConfig.model_training_path,obj=best_model)

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)