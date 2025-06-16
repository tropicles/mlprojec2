import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation , DataTransformationConfig


@dataclass
class DataingestionConfig:
    train_data:str=os.path.join('artifacts',"train.csv")
    test_data:str=os.path.join('artifacts',"test.csv")
    raw_data:str=os.path.join('artifacts',"raw.csv")


class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataingestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as df')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data,index=False,header=True)
            logging.info('Train test split is done')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data,index=False,header=True)
            logging.info('Data ingestion is completed')
            return(
                self.ingestion_config.train_data,
                self.ingestion_config.test_data,
            )
        except Exception as e:
            raise CustomeException(e,sys)

if __name__=="__main__":
    obj=Dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)