import sys
import os
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

#columntransformer is used to create pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from exception import CustomException
from logger import logging
from utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path=os.path.join('artifact',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:

            '''
            
            Data transformation will be carried out by this function

            Scope for improvement
            automate finding columns
            determine the pipeline creation with more options
            
            '''
            # can we automate the finding of numerical columns and categorical columns instead of hardcoding it
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            ## cleaning the data pipeline which is numerical columns
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))

                ]
            )

            # cleaning the data pipeline which is categorical columns
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info("Categorical Column encodings completed, Numerical Column standard scaling completed, Imputing carried out on both")

            preprocessor=ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):


        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            

            # how to automated this to build ML for any column programatically.
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
 
            #here two sets of data frames are created for training and testing, one being your target and the other being the inputs to be used for getting the input
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
                ]

            logging.info(f"Saved preprocessing object.")

            # responsible for saving the preprocessor pickle file
            save_object(

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)