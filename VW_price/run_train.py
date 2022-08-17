"""Module for running the training procedure"""

import argparse

from modules.data_preparation import TrainDataPreparator
from modules.model_factory import ModelFactory, LassoFactory, RandomForestFactory, GradientBoostingFactory

def read_model_type(model_type: str) -> ModelFactory:
    """Creates a model instance given a user selection"""
    model_types = {
        "lasso": LassoFactory(),
        "rf": RandomForestFactory(),
        "boosting": GradientBoostingFactory()
    }
    return model_types[model_type]


def run_training(model_factory: ModelFactory):
    """Main function for running the training procedure"""

    data, target = TrainDataPreparator().prepare_data()
    model = model_factory.get_model()
    model.prepare_model()
    model.train_model(data, target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lasso', 'rf', 'boosting'], default='rf')
    args = parser.parse_args()
    model_factory = read_model_type(args.model)
    run_training(model_factory)
