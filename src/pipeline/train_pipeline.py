# write the logic to build the training pipeline

import sys
import pandas as pd
from pathlib import Path
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import src.components.data_ingestion as di
import src.components.data_transformation as dt
import src.components.model_trainer as mt

sys.path.append(str(Path(__file__).parent.parent))

from logger import logging
from exception import CustomException
from utils import load_object


class TrainPipeline:
    def __init__(self,target_column_name: str):
        self.tcm = target_column_name

    def train_model(self):
        try:
            data_path=os.path.join("uploads","processed_data.csv")
            #preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            logging.info("passing file path")
            #preprocessor=load_object(file_path=preprocessor_path)
            obj = di.DataIngestion()
            train_data,test_data=obj.initiate_data_ingestion(data_path)
            logging.info("Here now two data sets should have been created for training and testing")
            #df=pd.read_csv(train_data)
            logging.info(f"The target column name being passed in the train_model is {self.tcm}")
            obj_dt=dt.DataTransformation(self.tcm)
            logging.info("data transformation is called")
            train_arr,test_arr,_= obj_dt.initiate_data_transformation(train_data,test_data)

            obj_mt = mt.ModelTrainer()
            result=obj_mt.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
    
            return result
        
        except Exception as e:
            raise CustomException(e,sys)




######################################### the pipeline will call the data_ingestion 
# available options
# location of where csv file is online as link
# upload a csv file

# this will call the data_ingestion component

# based on the read data we must show the headers of all the columns
# ask below questions
# columns to drop if any (non relevant)
# columns to modify if any (refactor) (use of AI)
# columns to add based on existing data (new columns) (use of AI)

############## > then the generated file is split into training and testing data set


######################################## the pipleine will call the data transformation

# will print out the numerical and categorical columns 
# will provide basic statistical information for the generated data 
# will perform cleaning of data
# will perform imputation, one hot encoding for categorical, scaling for all data


############# > the this will generate the preprocessor file

######################################## the pipleline will call the model trainer

# print the various training and test results
# provide the best model
# create the file of the best model


############ >  this will generate the model file


# return results

