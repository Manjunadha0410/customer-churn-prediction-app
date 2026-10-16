import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import ProjectError
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance']
            categorical_columns = ['Geography', 'Gender']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median"))
                ]
            )
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(drop="first"))
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline ,numerical_columns),
                    ("cat",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise ProjectError(e,sys)
        
    def starting_data_transformation(self,training_file_path,test_file_path):

        try:
            logging.info("Reading training and test data")
            train_df=pd.read_csv(training_file_path)
            test_df=pd.read_csv(test_file_path)

            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Preprocesser object obtained")
            target_column="Multi_product"

            input_train_df=train_df.drop(columns=[target_column])
            target_train_df=train_df[target_column]

            input_test_df=test_df.drop(columns=[target_column])
            target_test_df=test_df[target_column]

            input_train_arr=preprocessing_obj.fit_transform(input_train_df)
            input_test_arr=preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            logging.info(f"Saving preprocessor.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise ProjectError(e,sys)
