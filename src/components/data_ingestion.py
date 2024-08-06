## The component is responsible to get data from sources

import os
import sys
# below two lines were added to solve the src not found issue
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))



from logger import logging
from exception import CustomException
import pandas as pd


from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from data_transformation import DataTransformation, DataTransformationConfig
from model_trainer import ModelTrainerConfig, ModelTrainer


# SHOULD WE ALSO CREATE A VALIDATION DATA SET ?
#any input required is handled by the DataIngestionConfig , extend this in future to support api_data, folder_data etc
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    #currently it expects the data to be present in  a particular file called stud.csv
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Imported the csv data file into dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train and Test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion completed and train, test files saved')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))