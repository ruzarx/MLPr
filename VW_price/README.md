# Volkswagen Golf used car price prediction

## Description

The project arised when I was selling my car and tried to set adequate price for my advertisement. It takes car parameters and predicts its market price.

To run training:

```bash
python run_train.py --model lasso
or
python run_train.py --model rf
or
python run_train.py --model boosting
```

To run prediction:

```bash
python run_predict.py 
        --model [model_type] 
        --data_path Optional[path to dataset] 
        --manual_data Optional[boolean]
```

In case of manual data input selection the script will offer you to input the data manually (just follow the question which will follow).

## Data

Data was collected manually from Auto.ru. The sample consists of Volkswagen Golf cars.

Features:
- Car production year
- Mileage, km
- Engine volume: 1.2, 1.4, 1.6, 2.0
- Gearbox type: A (automatic) or M (manual)
- Car owner: a person (1) or a company (0)
- Car has 3 doors (1) or 5 doors (0)

Target:
- Expected price, RUR

## Overview

The program has two run modes: training and prediction (inference). The training script takes a user defined model type (lasso, random forest or gradient boosting), loads training data, trains and saves the model. The prediction script takes a model type and data and creates a model prediction for it. The prediction script can take either a file with the data or provide manual data input from console.

### Training pipeline

Training pipeline consists of the following stages:

- Reads model type from the console upon the script run (lasso, random forest or gradient boosting)
- Creates a model object from a model factory based on the provided model type
- Runs TrainDataPreparator for data preparation:
    - Loads training dataset
    - Filters unnecessary examples based on configuration parameters
    - Checks there are no zero values in the data
    - Encodes categorical features (engine volume, gearbox type and production year)
    - Normalizes continuous features (mileage and price)
    - Separates target from the rest of the dataset
    - Drops unnecessary columns
    - Saves the normalizers parameters
- Runs model training:
    - Creates a cross-validation setup based on configuration parameters
    - For each fold:
        - Trains a model on train subset
        - Making prediction on test subset
        - Reverting normalization for the predictions
        - Calculates metrics on test subset and stores into metrics dataclass
    - Visualizes quality metrics
    - Saves the model

### Inference pipeline

Inference pipeline consists of the following stages:

- Reads model type from the console upon the script run (lasso, random forest or gradient boosting)
- Reads either a path to inference dataset or offers manual console data inputation
- Creates a model object from a model factory based on the provided model type
- Loads a trained model (throws exception if model is not trained)
- Runs InferenceDataPreparator for data preparation:
    - Loads training dataset
    - Checks there are no zero values in the data
    - Encodes categorical features (engine volume, gearbox type and production year)
    - Normalizes continuous features (mileage) given previously saved scaler parameters
    - Drops unnecessary columns
- Runs model prediction:
    - Making prediction on a given dataset
    - Reverting normalization for the predictions
- Outputs the prediction results