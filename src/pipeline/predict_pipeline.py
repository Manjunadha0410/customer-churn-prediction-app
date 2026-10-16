import os
import sys
import pandas as pd
from src.exception import ProjectError
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_processed=preprocessor.transform(features)
            pred=model.predict(data_processed)
            return pred
        
        except Exception as e:
            raise ProjectError(e,sys)



class CustomData:
    def __init__(  self,
        CreditScore: int,
        Geography: str,
        Gender: str,
        Age: int,
        Tenure: int,
        Balance: float):

        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise ProjectError(e, sys)

