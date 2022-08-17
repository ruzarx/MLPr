"""Module for model selection factory"""

from abc import ABC, abstractmethod
from modules.model import Model, LassoModel, RandomForestModel, GradientBoostingModel

class ModelFactory(ABC):
    """Model factory class representing specific model types"""

    @abstractmethod
    def get_model(self) -> Model:
        """Returns a model instance"""

class LassoFactory(ModelFactory):
    """Factory providing a lasso model"""

    def get_model(self) -> Model:
        return LassoModel()

class RandomForestFactory(ModelFactory):
    """Factory providing a random forest model"""

    def get_model(self) -> Model:
        return RandomForestModel()

class GradientBoostingFactory(ModelFactory):
    """Factory providing a gradient boosting model"""

    def get_model(self) -> Model:
        return GradientBoostingModel()