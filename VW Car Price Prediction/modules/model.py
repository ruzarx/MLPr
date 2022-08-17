"""Module for training, trainin validation and prediction"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import pathlib
import json, pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, BaseShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from config import config
from modules.metrics_dataclass import MetricsStorage

class Model(ABC):
    """Abstract class for model"""

    def __init__(self):
        self.cv: BaseShuffleSplit
        self.metrics: MetricsStorage = MetricsStorage()

    @abstractmethod
    def prepare_model(self) -> None:
        """Prepares everything for training"""

    def train_model(self, train_data: pd.DataFrame, target: pd.Series) -> None:
        """Runs training procedure"""
        for train_index, test_index in self.cv.split(train_data, target):
            split_train_data = train_data.iloc[train_index]
            split_test_data = train_data.iloc[test_index]
            split_train_target = target[train_index]
            split_test_target = target[test_index]
            self.model.fit(split_train_data, split_train_target)
            test_prediction = self.model.predict(split_test_data)
            inverted_prediction = self.__invert_transform(test_prediction)
            inverted_target = self.__invert_transform(split_test_target)
            mae, mape, mse = self.calculate_metrics(inverted_prediction, inverted_target)
            self.metrics.add_fold_metrics(mae, mape, mse)
        self.metrics.depict_metrics()
        self.save_model()
        return

    def predict_model(self, prediction_data: pd.DataFrame) -> np.ndarray:
        """ Runs prediction procedure"""
        self.load_model()
        prediction = self.model.predict(prediction_data)
        prediction = self.__invert_transform(prediction)
        return prediction

    def __invert_transform(self, prediction: np.ndarray) -> np.ndarray:
        """Brings prediction back to non scaled values range"""
        scaler_params = self.__load_scaler()
        mean_value, std_value = scaler_params[config.data_config.target_column]
        inversed_prediction = prediction * std_value + mean_value
        return inversed_prediction

    def setup_cross_validation(self):
        self.cv = ShuffleSplit(n_splits=config.data_config.n_cv_splits)

    def calculate_metrics(self, prediction: List[float], target: pd.Series) -> Tuple[float, float, float]:
        """Calculates a fold metrics"""
        mae = mean_absolute_error(target, prediction)
        mape = mean_absolute_percentage_error(target, prediction)
        mse = mean_squared_error(target, prediction)
        return mae, mape, mse

    def __load_scaler(self) -> Dict[str, Tuple[float, float]]:
        """Load scalers parameters"""
        scaler_path = pathlib.Path(config.data_config.scalers_path)
        if scaler_path.is_file():
            with open(scaler_path, 'r') as load_file:
                scaler_params = json.load(load_file)
        else:
            raise FileNotFoundError(f"Scalers data file is not found there: {scaler_path}. Run training first")
        return scaler_params

class LassoModel(Model):
    """Lasso implementation for a Model class"""

    def prepare_model(self) -> None:
        """Prepares everything for training"""
        self.setup_cross_validation()
        self.model = Lasso(**config.model_params_config.lasso_params)
        return

    def save_model(self) -> None:
        """Saves trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path / 'model_lasso.pkl', 'wb') as save_file:
            pickle.dump(self.model, save_file)
        return

    def load_model(self) -> None:
        """Loads trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        with open(model_path / 'model_lasso.pkl', 'rb') as load_file:
            self.model = pickle.load(load_file)
        return


class RandomForestModel(Model):
    """RandomForest implementation for a Model class"""

    def prepare_model(self) -> None:
        """Prepares everything for training"""
        self.setup_cross_validation()
        self.model = RandomForestRegressor(**config.model_params_config.random_forest_params)
        return

    def save_model(self) -> None:
        """Saves trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path / 'model_rf.pkl', 'wb') as save_file:
            pickle.dump(self.model, save_file)
        return

    def load_model(self) -> None:
        """Loads trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        with open(model_path / 'model_rf.pkl', 'rb') as load_file:
            self.model = pickle.load(load_file)
        return

class GradientBoostingModel(Model):
    """Gradient boosting implementation for a Model class"""

    def prepare_model(self) -> None:
        """Prepares everything for training"""
        self.setup_cross_validation()
        self.model = LGBMRegressor(**config.model_params_config.boosting_params)
        return

    def save_model(self) -> None:
        """Saves trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path / 'model_boosting.pkl', 'wb') as save_file:
            pickle.dump(self.model, save_file)
        return

    def load_model(self) -> None:
        """Loads trained model"""
        model_path = pathlib.Path(config.data_config.model_path)
        with open(model_path / 'model_boosting.pkl', 'rb') as load_file:
            self.model = pickle.load(load_file)
        return
