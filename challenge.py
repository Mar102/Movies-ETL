#!/usr/bin/env python
# coding: utf-8

# In[1]:



import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import psycopg2
import time
import sqlalchemy


# In[2]:


# Define a variable for the directory holding the data.
file_dir = 'C:/Users/avilam2/Desktop/DataAustin2020/Module_8/'
f'{file_dir}filename'


# In[3]:


###### Extract the data ########
# Create a function that takes in three arguments: Wikipedia data, Kaggle metadata, MovieLens rating data (from Kaggle)
def Wiki_Kaggle_Ratings(wiki_movies_1, kaggle_metadata, ratings):
    
    #Open Wikipedia file
    with open (f'{file_dir}/wikipedia.movies.json', mode='r') as file:
        wiki_movies_1 = json.load(file)
    
    # open kaggle file
    kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory=False)  

    #Open Movielens rating file
    ratings = pd.read_csv(f'{file_dir}ratings.csv')

    
   ###### Transforming ########

    # Inspect Wiki_data
    wiki_movies = [movie for movie in wiki_movies_1
                   if ('Director' in movie or 'Directed by' in movie)
                       and 'imdb_link' in movie
                       and 'No. of episodes' not in movie]

    # Convert wiki_movies_1 into a DF
    wiki_movies_df = pd.DataFrame(wiki_movies_1)

    #Sort columns
    sorted(wiki_movies_df.columns.tolist())

    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune-Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles

        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

    #Rerun list comprehension to clean wiki_movies and recreate wiki_movies_df
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)
    try:
        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
        wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    except:
        print(f'"encountered an error....proceeding"')



    # Check how many null values are in each column and keep any that are less than 90% null 
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    #Convert and Parse box office column 
    box_office = wiki_movies_df['Box office'].dropna() 
    # Make a separator string using apply method and then call the join() method on it.
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    #Preface the string with an r for the escape characters to remain.
    form_one = r'\$\d+\.?\d*\s*[mb]illion'
    box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

    # Preface the string with an r for the escape characters to remain.
    form_two = r'\$\d{1,3}(?:,\d{3})+'
    box_office.str.contains(form_two, flags=re.IGNORECASE).sum()
    
    # Create two Boolean Series called matches_form_one and matches_form_two, and then select the box office values that don’t match either
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
    box_office[~matches_form_one & ~matches_form_two]

    # Now we need a function to turn the extracted values into a numeric value. We’ll call it parse_dollars
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

    #Parse box office values
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)      [0].apply(parse_dollars)

    # Drop the original box office column
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    
    # Create a budget variable 
    budget = wiki_movies_df['Budget'].dropna()

    # Convert any lists to strings
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

    # Remove any values between a dollar sign and a hyphen (for budgets given in ranges)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # Create two Boolean Series called matches_form_one and matches_form_two, 
    # and then select the budget values that don’t match either
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget[~matches_form_one & ~matches_form_two]

    # Remove the citation references.
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]

    # parse the budget values.
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0]       .apply (parse_dollars)

    # Drop the original Budget columns.
    wiki_movies_df.drop('Budget', axis=1, inplace=True)


    # Make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings:
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # One way to parse those forms is with the following:
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    # Extract the dates
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

    # Use the to_datetime() method to set the infer_datetime_format option to True.
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # Parse Running Time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # Save extracted running time and convert to nans to 0. coerce makes errors NaNs. fillna() makes NaNs 0.
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    # Use apply() to convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero, and save the output to wiki_movies_df
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    #Drop Running time from the dataset
    wiki_movies_df.drop('Running time', axis=1, inplace=True)


    #CLEAN THE KAGGLE DATA
    # Remove the bad data
    kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

    #keep rows where the adult column is False and drop the adult column.
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # Convert and remove non-video from the kaggle DataFrame
    kaggle_metadata['video'] == 'True'
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # Use the to_numeric() method from Pandas to convert colums, errors= argument is set to 'raise', to know if there’s any data that can’t be converted to numbers.
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

    # Convert release_date to datetime.
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # Reasonability Checks on Ratings Data

    # Convert timestamp to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Merge DataFrames to identify redundant columns
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])   

    # We should investigate that wild outlier around 2006. We’re just going to choose some rough cutoff dates to single out that one movie. We’ll look for any movie whose release date according to Wikipedia is after 1996, but whose release date according to Kaggle is before 1965.
    movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]

    # Get the index.
    movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

    # Drop the row
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # Drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # Create a function that fills in missing data for a column pair and then drops the redundant column.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # Run the function for the three column pairs that we decided to fill in zeros.
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # Reorder the columns 
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                           'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                           'genres','original_language','overview','spoken_languages','Country',
                           'production_companies','production_countries','Distributor',
                           'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                          ]]

    # Rename the columns to be consistent.
    movies_df.rename({'id':'kaggle_id',
                      'title_kaggle':'title',
                      'url':'wikipedia_url',
                      'budget_kaggle':'budget',
                      'release_date_kaggle':'release_date',
                      'Country':'country',
                      'Distributor':'distributor',
                      'Producer(s)':'producers',
                      'Director':'director',
                      'Starring':'starring',
                      'Cinematography':'cinematography',
                      'Editor(s)':'editors',
                      'Writer(s)':'writers',
                      'Composer(s)':'composers',
                      'Based on':'based_on'
                     }, axis='columns', inplace=True)

    # TRANSFORM AND MERGE RATING DATA

    #Rename the “userId” column to “count and ”pivot this data so that movieId is the index, the columns will be all the rating values, and the rows will be the counts for each rating value.
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                     .rename({'userId':'count'}, axis=1)                     .pivot(index='movieId',columns='rating', values='count')

    # Rename the columns so they’re easier to understand.
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    # Use a left merge, since we want to keep everything in movies_df:\
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # Replace NaNs with 0s Since not every movie got a rating for each rating level, there will be missing values instead of zeros.
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # LOAD THE DATA
    # Make a connection string for the local server.
    db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data"

    # Create the database engine 
    engine = create_engine(db_string)

    # Delete movies table content in sql
    movie_data = engine.connect()
    truncate_query = sqlalchemy.text("TRUNCATE TABLE movies")
    movie_data.execution_options(autocommit=True).execute(truncate_query)
    print('Removed content in movies sql table')

    # Delete ratings table content in sql
    movie_data = engine.connect()
    truncate_query = sqlalchemy.text("TRUNCATE TABLE ratings")
    movie_data.execution_options(autocommit=True).execute(truncate_query)
    print('Removed content in ratings sql table')

    # Use the to_sql() method to save the movies_df DataFrame to a SQL table
    movies_df.to_sql(name='movies', con=engine, if_exists='append')

    # Save ratings to SQL
    rows_imported = 0

    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

        # print out the range of rows that are being imported
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')


# In[4]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop Wikipedia
# running_time             runtime                  Keep Kaggle; fill in zeros with Wikipedia data.
# budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with Wikipedia data.
# box_office               revenue                  Keep Kaggle; fill in zeros with Wikipedia data.
# release_date_wiki        release_date_kaggle      Drop Wikipedia.
# Language                 original_language        Drop Wikipedia.
# Production company(s)    production_companies     Drop Wikipedia.


# In[5]:


# Exporting the files 
Wiki_Kaggle_Ratings("wikipedia.movies.json", "movies_metadata.csv", "ratings.csv")


# In[ ]:





# In[ ]:





# In[ ]:




