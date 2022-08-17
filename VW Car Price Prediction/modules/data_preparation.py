"""Module for input data preparation for training"""

from abc import ABC, abstractmethod
import pathlib
from typing import Optional, List, Dict, Tuple
import json

import pandas as pd
import numpy as np

from config import config

class DataPreparator(ABC):
    """Abstract class for input data prepraration for training and prediction"""

    def __init__(self):
        self.columns_to_drop: List[str] = ['Id']
        self.scaler_params: Dict[str, Tuple[float, float]] = {}

    @abstractmethod
    def prepare_data(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """The main method for data preparation"""

    @abstractmethod
    def scale_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical columns data"""

    def load_dataset(self, user_data_path: Optional[str] = None, manual_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Loads a dataset either from a user provided path (for inference)
        or default training dataset path
        """
        if manual_data is not None:
            return manual_data
        elif user_data_path is None:
            data_path = pathlib.Path(config.data_config.train_data_path)
        else:
            data_path = pathlib.Path(user_data_path)
        if data_path.is_file():
            dataset = pd.read_csv(data_path) #type: ignore
            return dataset
        else:
            raise FileNotFoundError(f"Data file is not found there: {data_path}")

    def filter_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Filters out irrelevant data from the dataset"""
        for col in ['Price', 'Mileage']:
            self.check_column_presence(dataset, col)
        # Dropping suspiciously cheap cars
        dataset = dataset[dataset['Price'] > int(config.data_config.cheap_car_threshold)]
        # Droppong with too high mileage
        dataset = dataset[dataset['Mileage'] < int(config.data_config.overused_car_threshold)]
        dataset = dataset.reset_index(drop=True)
        return dataset

    def check_zero_values(self, dataset: pd.DataFrame) -> None:
        """Checks if dataset does not contain zero values where not necessary"""
        non_zero_columns = [column for column in dataset.columns if column not in config.data_config.binary_columns]
        for column in non_zero_columns:
            column_zero_values = dataset[dataset[column] == 0].shape[0]
            if column_zero_values > 0:
                msg = f"Column {column} is not expected to have zero values, but does have {column_zero_values}"
                raise ValueError(msg)

    def encode_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process encoding of all columns, which need encoding"""
        dataset = self.encode_year(dataset)
        dataset = self.encode_gearbox(dataset)
        dataset = self.encode_engine_volume(dataset)
        return dataset

    def drop_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns"""
        drop_cols = [column for column in self.columns_to_drop if column in dataset.columns]
        dataset_dropped = dataset.drop(columns=drop_cols)
        return dataset_dropped

    def check_column_presence(self, dataset: pd.DataFrame, column: str) -> None:
        if column not in dataset.columns:
            raise KeyError(f"Column {column} not found in dataset")

    def encode_year(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Encode production year to car age"""
        self.check_column_presence(dataset, 'Year')
        production_year = dataset['Year'].astype(int).values
        current_year = 2020
        car_age = current_year - production_year
        dataset['Age'] = car_age
        self.columns_to_drop += ['Year']
        return dataset

    def encode_gearbox(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Encode gearbox type to binary feature"""
        self.check_column_presence(dataset, 'Gearbox')
        dataset['automatic_gearbox'] = 0
        dataset.loc[dataset['Gearbox'] == 'A', 'automatic_gearbox'] = 1
        self.columns_to_drop += ['Gearbox']
        return dataset

    def encode_engine_volume(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Encode engine volume to binary features"""
        self.check_column_presence(dataset, 'Volume')
        dataset['Volume'] = dataset['Volume'].astype(float)
        for value in [1.2, 1.4, 1.6, 2.0]:
            dataset[f"Volume_{value}"] = 0
            dataset.loc[dataset['Volume'] == value, f"Volume_{value}"] = 1
        self.columns_to_drop += ['Volume']
        return dataset

    def scale_new_column(self, dataset: pd.DataFrame, column: str) -> pd.DataFrame:
        """Scale column for training dataset"""
        mean_value = dataset[column].mean()
        std_value = dataset[column].std()
        self.scaler_params[column] = (mean_value, std_value)
        dataset[column] = (dataset[column] - mean_value) / std_value
        return dataset

    def scale_existing_column(self, dataset: pd.DataFrame, column: str) -> pd.DataFrame:
        """Scale column with existing parameters for prediction"""
        mean_value, std_value = self.scaler_params[column]
        dataset[column] = (dataset[column] - mean_value) / std_value
        return dataset


class TrainDataPreparator(DataPreparator):
    """DataPreparator class for training data preparation"""

    def prepare_data(self,
                     data_path: Optional[str] = None,
                     manual_data: Optional[pd.DataFrame] = None
                     ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """The main method for data preparation"""
        dataset = self.load_dataset()
        dataset = self.filter_data(dataset)
        self.check_zero_values(dataset)
        dataset = self.encode_columns(dataset)
        dataset = self.scale_columns(dataset)
        target = self.__separate_target(dataset)
        dataset = self.drop_columns(dataset)
        self.__save_scaler()
        return (dataset, target)

    def scale_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical columns data"""
        for column in config.data_config.numerical_columns + [config.data_config.target_column]:
            dataset = self.scale_new_column(dataset, column)
        return dataset

    def __separate_target(self, dataset: pd.DataFrame) -> pd.Series:
        """Separates target from the main data"""
        self.check_column_presence(dataset, 'Price')
        target = dataset['Price'].astype(float)
        self.columns_to_drop += ['Price']
        return target

    def __save_scaler(self) -> None:
        """Saves scalers parameters"""
        scaler_path = pathlib.Path(config.data_config.scalers_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'w') as save_file:
            json.dump(self.scaler_params, save_file)


class InferencesDataPreparator(DataPreparator):
    """DataPreparator class for inference data preparation"""

    def __init__(self):
        super().__init__()
        self.__load_scaler()

    def prepare_data(self,
                     data_path: Optional[str] = None,
                     manual_data: Optional[pd.DataFrame] = None
                     ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """The main method for data preparation"""
        self.columns_to_drop += [config.data_config.target_column]
        dataset = self.load_dataset(data_path, manual_data)
        self.check_zero_values(dataset)
        dataset = self.encode_columns(dataset)
        dataset = self.drop_columns(dataset)
        dataset = self.scale_columns(dataset)
        return dataset, None

    def scale_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical columns data"""
        for column in config.data_config.numerical_columns:
            dataset = self.scale_existing_column(dataset, column)
        return dataset

    def __load_scaler(self) -> None:
        """Load scalers parameters"""
        scaler_path = pathlib.Path(config.data_config.scalers_path)
        if scaler_path.is_file():
            with open(scaler_path, 'r') as load_file:
                self.scaler_params = json.load(load_file)
        else:
            raise FileNotFoundError(f"Scalers data file is not found there: {scaler_path}. Run training first")
