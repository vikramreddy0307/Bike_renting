# Bike_renting


 # Introduction						
* Problem statement:
    * The objective of this Case is the Predication of bike rental count on daily based on the
        environmental and seasonal settings.
    * I have performed data analysis on the factors that are affecting total count of the variable and designed a model which       predicts the total count and made a complete report of the project.

* Data
    The details of data attributes in the dataset are as follows :

    * instant: Record index
    * dteday: Date
    * season: Season (1:springer, 2:summer, 3:fall, 4:winter)
    * yr: Year (0: 2011, 1:2012)
    * mnth: Month (1 to 12)
    * holiday: weather day is holiday or not (extracted fromHoliday Schedule)
    * weekday: Day of the week
    * workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
    * weathersit: (extracted fromFreemeteo)
        *  Clear, Few clouds, Partly cloudy, Partly cloudy
        *  Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        *  Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered
        clouds
        *  Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    * temp: Normalized temperature in Celsius. 
    * atemp: Normalized feeling temperature in Celsius. 
    * hum: Normalized humidity. 
    * windspeed: Normalized wind speed. 
    * casual: count of casual users
    * registered: count of registered users
    * cnt: count of total rental bikes including both casual and registered

# Exploratory data analysis
* Recoding factor variables
* Data exploration
* Histogram distribution  of continuous variables
* Bar distribution of categorical variables
* Distribution of categorical variables across total count
* Distribution of categorical variables across casual and registered users
* Boxplot graphs of variables

# Data preprocessing
* One hot encoding of factor variables
* Removing correlated features from the data

# Feature Engineering
* Creating new features
* Feature selection using Boruta in R
* Feature selection using seleck-K best in python
* Principal component analysis
# Sampling methods
* K-fold repeated CV
# Machine learning models
* Linear Regression
* Random forest
* Elastic net
* Lasso regression
* Ridge regression
# Model results
* Model results with feature selection in R and python
	* RMSE
	* R-squared
* Model results with PCA in R and python
	* RMSE
	* R-squared
* Comparing results of both the model
* Selecting the best fitted model

