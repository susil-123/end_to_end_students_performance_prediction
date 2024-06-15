import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    data_transformation_config_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()
    
    def get_data_tranformation_object(self):
        try:
            logging.info("Entered get_transformation_object")
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            
            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('standardscaler',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('standardscaler',StandardScaler(with_mean=False))
            ])

            preprocessor_obj = ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor_obj
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_data_path,test_data_path):
        try:
            logging.info("Entered into initiate_data_tranformation()")

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("read the train and test files")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_tranformation_object()
            target_column_name = 'math_score'

            X_train = train_data.drop(columns=[target_column_name])
            X_test = test_data.drop(columns=[target_column_name])
            
            y_train = train_data[target_column_name]
            y_test = test_data[target_column_name]
            logging.info("initialized xtrain, xtest, ytrain, ytest")

            X_train_preprocessed = preprocessor_obj.fit_transform(X_train)
            X_test_preprocessed  = preprocessor_obj.transform(X_test)
            logging.info("fitted the preprocessing object on to the data")

            train_data_preprocessed = np.c_[X_train_preprocessed,np.array(y_train)]
            test_data_preprocessed = np.c_[X_test_preprocessed,np.array(y_test)]

            save_obj(file_path=self.DataTransformationConfig.data_transformation_config_path,obj=preprocessor_obj)
            logging.info("saved the preprocessor object in the directory")

            return (
                train_data_preprocessed,
                test_data_preprocessed,
                self.DataTransformationConfig.data_transformation_config_path
            )

        except Exception as e:
            raise CustomException(e,sys)
