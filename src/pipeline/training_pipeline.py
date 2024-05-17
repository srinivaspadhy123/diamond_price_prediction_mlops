import os
import sys
from src.logger.logging import logging
from src.exceptions.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

obj = DataIngestion()
train_data_path,test_data_path = obj.initiate_data_ingesiton()

data_transformation = DataTransformation()
train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)
model_evaluation_obj = ModelEvaluation()
model_evaluation_obj.initiate_model_evaluation(train_arr,test_arr)