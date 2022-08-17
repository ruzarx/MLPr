"""Module for describing a dataclass for train metrics representation"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

@dataclass
class MetricsStorage:
    folds: int = field(init=False, default=0)
    maes: List[float] = field(default_factory=list)
    mapes: List[float] = field(default_factory=list)
    mses: List[float] = field(default_factory=list)

    def add_fold_metrics(self, mae: float, mape: float, mse: float) -> None:
        self.folds += 1
        self.maes += [mae]
        self.mapes += [mape]
        self.mses += [mse]
        return

    def depict_metrics(self) -> None:
        if self.folds == 0:
            print("No metrics to show, traing the model first")
            return
        result = pd.DataFrame(columns=['Metric', 'Fold minimum', 'Fold mean', 'Fold maximum'])
        for metric, name in zip([self.maes, self.mapes, self.mses], ['MAE', 'MAPE', 'MSE']):
            mean_value, max_value, min_value = self.__process_metric(metric_values=metric)
            result = pd.concat([result, pd.DataFrame({'Metric': [name],
                                                   'Fold minimum': [round(min_value, 2)],
                                                   'Fold mean': [round(mean_value, 2)],
                                                   'Fold maximum': [round(max_value, 3)]})])
        print(result)
        return

    def __process_metric(self, metric_values: List[float]) -> Tuple[float, float, float]:
        mean_value = sum(metric_values) / self.folds
        max_value = max(metric_values)
        min_value = min(metric_values)
        return mean_value, max_value, min_value
