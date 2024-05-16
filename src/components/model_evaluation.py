import os
import sys
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from src.utils.utils import evaluate_model,load_object
from src.logger.logging import logging
from src.exceptions.exception import CustomException
from dataclasses import dataclass

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as exp:
            logging.info()
            raise CustomException(exp,sys)



