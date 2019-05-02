# Data Science Projects

There are two major types of projects represented in this repo: mini-projects, dedicated to solving some minor problem, or data analysis; major projects with complex approach.

## Projects

### 1. Formula-1 Bet Predictor and Win Optimizer

<b>Motivation</b>. This project is inspired by my passion to Formula-1 racing. The goal was to predict probabilities of all the drivers to finish in top-10 in upcoming race and based on these probabilities and bookmaker's coefficients optimize the set and money bets to maximize the profit.

<b>Data</b>. Data was collected manually from F1.com and Wikipedia. Incudes all races since season 2015.

<b>Specifics</b>. Each race, consisting of 20 drivers, is a separate event, which has specific track, weather conditions, drivers' and cars' form and so on. So there is not that straight away approach to put all the available data in one dataframe. History of drivers' and cars' form were figured out.

Another snag is a big range of randomeness of race results due to relatively large probability of crashes, drivers' mistakes or car failures. There nuances were considered in probabilistic manner.

<b>Model</b>. The algorithm trained with LightGBM.

<b>Optimizer</b>. A manual profit optimizer was build based on <link href="https://en.wikipedia.org/wiki/Sharpe_ratio">Sharpe ratio</link> maximization, which is used in investment optimizations.

<b>Results</b>. Pure probabilistic model predicts correctly 9 out of 10 top finishers in average. Profit optimizator makes about 20% profit for a race in average.

### 2. Medical Diagnostics Predictor (<link href="https://github.com/ruzarx/MLPr/blob/master/Transcriptions/Diagnostics%20prediction.ipynb">ipynb</link>, <link href="https://github.com/ruzarx/MLPr/blob/master/Transcriptions/Diagnostics%20prediction.html">html</link>)

<b>Motivation></b>. Preparation for my future project with a medic companion to build a ML NLP model, which will predict required diagnostics for a patient based on his/her anamnesis, written in free text form.

<b>Data</b>. Dataset consist of 5000 records of anamnesises (in average 150 words each) and required diagnostics (40 labels).

<b>Specifics</b>. Classes are vastly imbalanced (over 1000 records in a major one and less than 10 in minor). Different methods of oversampling (naive random cut, NearMiss) and undersampling (SMOTE, ADASYN) are used.

<b>Model</b>. Compared on Bag of Words, TF-IDF with 1, 2 and 3-grams, Word2Vec, using Naive Bayes classifier, LogisticRegression, CNN (not included in the report) and LightGBM models. Word2Vec was pretrained on common GoogleNews dataset and on specific medical articles dataset. Surprisingly the common one worked better.

<b>Results</b>. Model managed to achieve over 0.6 in F1-score and 0.55 in recall compared to baseline around 0.3 for both metrics.

### 3. Online User Identity

## Mini-projects

### 1. Car Price Prediction

### 2. Car Maintenance Price Prediction

### 3. Online Gaming Data Analysis
