# Data Science Projects

There are two major types of projects represented in this repo: mini-projects, dedicated to solving some minor problem, or data analysis; major projects with complex approach.

## Summary

### Projects

1. Formula-1 Bet Predictor and Win Optimizer - predicts probabilities of every F1 driver to finish in top-10 and based on these probabilities and bookmakers coefficients calculates the optimal bet for maximum expected win.
2. Medical Diagnostics Predictor - predicts diagnostics required for a patient based on his/her anamnesys.
3. Online User Identity - predicts specific internet user based on his/her behavior online.

### Mini-projects

1. Car Price Prediction - predicts used car market price based on Auto.ru ads and car characteristics.
2. Online Gaming Data Analysis - exploring the effect of online matching algorithm change.
3. Car Maintenance Price Analysis - car maintenance price analysis based on car model, producer and other characteristics.

## Projects

### 1. Formula-1 Bet Predictor and Win Optimizer

<b>Motivation</b>. This project is inspired by my passion to Formula-1 racing. The goal was to predict probabilities of all the drivers to finish in top-10 in upcoming race and based on these probabilities and bookmaker's coefficients optimize the set and money bets to maximize the profit.

<b>Data</b>. Data was collected manually from F1.com and Wikipedia. Incudes all races since season 2015.

<b>Specifics</b>. Each race, consisting of 20 drivers, is a separate event, which has specific track, weather conditions, drivers' and cars' form and so on. So there is not that straight away approach to put all the available data in one dataframe. History of drivers' and cars' form were figured out.

Another snag is a big range of randomeness of race results due to relatively large probability of crashes, drivers' mistakes or car failures. There nuances were considered in probabilistic manner.

<b>Model</b>. The algorithm trained with LightGBM.

<b>Optimizer</b>. A manual profit optimizer was build based on [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) maximization, which is used in investment optimizations.

<b>Results</b>. Pure probabilistic model predicts correctly 9 out of 10 top finishers in average. Profit optimizator makes about 20% profit for a race in average.

### 2. Medical Diagnostics Predictor ([ipynb](https://github.com/ruzarx/MLPr/blob/master/Transcriptions/Diagnostics%20prediction.ipynb), [html](https://github.com/ruzarx/MLPr/blob/master/Transcriptions/Diagnostics%20prediction.html))

<b>Motivation></b>. Preparation for my future project with a medic companion to build a ML NLP model, which will predict required diagnostics for a patient based on his/her anamnesis, written in free text form.

<b>Data</b>. Dataset consist of 5000 records of anamnesises (in average 150 words each) and required diagnostics (40 labels).

<b>Specifics</b>. Classes are vastly imbalanced (over 1000 records in a major one and less than 10 in minor). Different methods of oversampling (naive random cut, NearMiss) and undersampling (SMOTE, ADASYN) are used.

<b>Model</b>. Compared on Bag of Words, TF-IDF with 1, 2 and 3-grams, Word2Vec, using Naive Bayes classifier, LogisticRegression, CNN (not included in the report) and LightGBM models. Word2Vec was pretrained on common GoogleNews dataset and on specific medical articles dataset. Surprisingly the common one worked better.

<b>Results</b>. Model managed to achieve over 0.6 in F1-score and 0.55 in recall compared to baseline around 0.3 for both metrics.

### 3. Online User Identity ([ipynb](https://github.com/ruzarx/MLPr/blob/master/Online_User_Identity/User_identification.ipynb))

<b>Motivation</b>. Final project of Yandex/MIPT Coursera specialization. It is aimed to predict a specific internet user based on his/her browsing history.

<b>Data</b>. Source data is represented by users lists, webpages they visited and timestamps of the visit.

<b>Specifics</b>. As for every user browsing sessions are massively different in count (somebody visits dozens of pages in one session and someone visits only a few), the session window approach was used - each session was limited by 7, 10 or 15 visits (compared inside). All pages above that number for the same user were considered in another sessions. 

<b>Model</b>. Logistic regression worked well enough for the task.

<b>Results</b>. Model showed over 0.87 with ROC-AUC score.

## Mini-projects

### 1. Car Price Prediction ([ipynb](https://github.com/ruzarx/MLPr/blob/master/VW_price/VW_Price.ipynb))

<b>Motivation</b>. The project arised when I was selling my car and tried to set adequate price for my advertisement.

<b>Data</b>. Data was collected manually from Auto.ru. The sample consists of Volkswagen Golf cars.

<b>Model</b>. As the data looks a lot like linear, LassoCV model was chosen for fitting.

<b>Results</b>. The accuracy achieved on my exact car with the model differed by less than 1% compared to the price, calculated by Auto.ru algorithm.

### 2. Online Gaming Data Analysis ([ipynb](https://github.com/ruzarx/MLPr/blob/master/Game_analysis/Game_matching_algorithm_analysis.ipynb))

<b>Motivation</b>. Exploration of dataset of online shooter game statistics in order for analyse the quality of changes in gamers eperience due to players matching algorithm change.

<b>Data</b>. Dataset consists of match timestamp, teams distributions, players levels and match statistics - frags, assists and deaths.

<b>Specifics</b>. The data is specific due to online gaming features. So I conducted detailed analysis of the dataset and the effect of matching algorithm change.

### 3. Car Maintenance Price Analysis ([ipynb](https://github.com/ruzarx/MLPr/blob/master/Car_dataset/Car_Maintenance.ipynb))

<b>Motivation</b>. Exploration of car maintenance price based on different car characteristics including technical ones and common ones, like car producer, model and so on.
