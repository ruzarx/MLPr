"""Module for running the prediction procedure"""

import argparse
from typing import Tuple, Optional
import pandas as pd

from modules.data_preparation import InferencesDataPreparator
from modules.model_factory import ModelFactory, LassoFactory, RandomForestFactory, GradientBoostingFactory

def read_model_type(model_type: str) -> ModelFactory:
    """Creates a model instance given a user selection"""
    model_types = {
        "lasso": LassoFactory(),
        "rf": RandomForestFactory(),
        "boosting": GradientBoostingFactory()
    }
    return model_types[model_type]

def process_input_data(input_data: str, feature: str):
    """Processes user input for manual prediction data input"""
    DISCREET_STR_INPUTS = {'Volume': ['1.2', '1.4', '1.6', '2.0'],
                       'Gearbox': ['A', 'M']}
    DISCREET_INT_INPUTS = {'Private': ['0', '1'],
                           '3-doors': ['0', '1']}
    CONTINUOUS_INPUTS = {'Year': [2000, 2022],
                         'Mileage': [0, 1_000_000]}
    if feature in DISCREET_STR_INPUTS:
        if input_data in DISCREET_STR_INPUTS[feature]:
            return input_data
        else:
            print(f"Can't pass this value for {feature}. It should be in {DISCREET_STR_INPUTS[feature]}")
            return None
    elif feature in DISCREET_INT_INPUTS:
        if input_data in DISCREET_INT_INPUTS[feature]:
            return int(input_data)
        else:
            print(f"Can't pass this value for {feature}. It should be in {DISCREET_INT_INPUTS[feature]}")
            return None
    else:
        try:
            float(input_data)
        except ValueError:
            print(f"Can't pass this value for {feature}. It should be integer of float")
            return None
        if float(input_data) < CONTINUOUS_INPUTS[feature][0]:
            print(f"The value passed for {feature} is too low. It should be at least {CONTINUOUS_INPUTS[feature][0]}")
            return None
        elif float(input_data) > CONTINUOUS_INPUTS[feature][1]:
            print(f"The value passed for {feature} is too high. It should be at most {CONTINUOUS_INPUTS[feature][1]}")
            return None
        return float(input_data)
    return

def manual_data_input() -> pd.DataFrame:
    """Procedure for manual data inputation"""
    print("Input prediction data manually")
    features = ['Year', 'Mileage', 'Volume', 'Gearbox', 'Private', '3-doors']
    messages = ['the car production year (2000-2022)',
                'the car mileage (0-1M km)',
                'the car engine volume (1.2, 1.4, 1.6, 2.0)',
                'the cat gearbox type (A, M)',
                'is the car is personal (0, 1)',
                'if the car has 3 doors (0, 1)']
    dataset = {}
    for feature, message in zip(features, messages):
        input_data = None
        while input_data is None:
            raw_input_data = input(f"Enter {message}: ")
            input_data = process_input_data(raw_input_data, feature)
        dataset[feature] = input_data
    dataset = pd.DataFrame(dataset, index=[0])
    return dataset
        

def read_data_parameters(data_path: str, manual_data: bool) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Parses the prediction data input parameters"""
    input_data_path = None
    input_manual_data = None
    if data_path == '':
        if not manual_data:
            raise ValueError("You must pass either path to data file or select manual data input")
        else:
            input_manual_data = manual_data_input()
    else:
        input_data_path = data_path
    return input_data_path, input_manual_data



def run_prediction(model_factory: ModelFactory, data_path: str, manual_data: pd.DataFrame):
    """Main function for running the training procedure"""

    data, _ = InferencesDataPreparator().prepare_data(data_path=data_path, manual_data=manual_data)
    model = model_factory.get_model()
    model.prepare_model()
    preds = model.predict_model(data)
    print(f"\nPredicted price: {round(preds[0], 2)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lasso', 'rf', 'boosting'], default='rf')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--manual_data', type=bool, default=False)
    args = parser.parse_args()

    data_path, manual_data = read_data_parameters(args.data_path, args.manual_data)
    model_factory = read_model_type(args.model)
    run_prediction(model_factory, data_path, manual_data)
