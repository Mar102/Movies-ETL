# Movies-ETL

## Project Objective ## 
 * Create an automated ETL pipeline.
 * Extract data from multiple sources.
 * Clean and transform the data automatically using Pandas and regular expressions.
 * Load new data into PostgreSQL.

## Resources ## 
Data Source: wikipedia-movies.json, movies_metadata.csv, ratings.csv
Software: Python 3.7.6, Anaconda 4.8.3, Jupyter Notebook 6.0.3, PostgreSQL 11.8 (pgAdmin 4.23)

## Challenge Overview ## 
* Create a function that takes in three arguments:
  * Wikipedia data
  * Kaggle metadata
  * Kaggle rating data (from Kaggle)


* Use the code from your Jupyter Notebook so that the function performs all of the transformation steps. Remove any exploratory data analysis and redundant code.
* Add the load steps from the Jupyter Notebook to the function. You’ll need to remove the existing data from SQL, but keep the empty tables.
* Check that the function works correctly on the current Wikipedia and Kaggle data.
* Document any assumptions that are being made. Use try-except blocks to account for unforeseen problems that may arise with new data.

## Assumptions ## 
* There is a lot of assumptions made because the challenge file is missing a lot of the explanatory data that you will need to easily understand the code.
* When accounting for Runtime, seconds are not being considered. The framework only includes hours, minites and "hour & minutes"
* Dropped the Wikipedia data because Wikipedia data is missing release dates for 11 movies, which are all in the Kaggle data. The lack of consistency from the Wiki data also made it hard to translate to the same format for analysis, providing more reasoning to drop it. 
* Replaced NaN with zero's to better utilize dataset.
