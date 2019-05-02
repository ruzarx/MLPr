# Data Science Projects

There are two major types of projects represented in this repo: mini-projects, dedicated to solving some minor problem, or data analysis; major projects with complex approach.

## Projects

### 1. Formula-1 Bet Predictor and Win Optimizer

<b>Motivation</b>. This project is inspired by my passion to Formula-1 racing. The goal was to predict probabilities of all the drivers to finish in top-10 in upcoming race and based on these probabilities and bookmaker's coefficients optimize the set and money bets to maximize the profit.

<b>Data</b>. Data was collected manually from F1.com and Wikipedia.

<b>Specifics</b>. Each race, consisting of 20 drivers, is a separate event, which has specific track, weather conditions, drivers' and cars' form and so on. So there is not that straight away approach to put all the available data in one dataframe. History of drivers' and cars' form were figured out.
Another snag is a big range of randomeness of race results due to relatively large probability of crashes, drivers' mistakes or car failures. There nuances were considered in probabilistic manner.

<b>Model</b>. The algorithm trained with LightGBM.

<b>Optimizer</b>. A manual profit optimizer was build based on <link href="https://en.wikipedia.org/wiki/Sharpe_ratio">Sharpe ratio</link> maximization, which is used in investment optimizations.

<b>Results</b>. Pure probabilistic model predicts correctly 9 out of 10 top finishers in average. Profit optimizator makes about 20% profit for a race in average.

### 2. Medical Diagnostics Predictor

### 3. Online User Identity

## Mini-projects

### 1. Car Price Prediction

### 2. Car Maintenance Price Prediction

### 3. Online Gaming Data Analysis
