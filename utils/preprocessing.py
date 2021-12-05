import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import numpy as np


def clean_reviews(review):
    clean_review = BeautifulSoup(review, features='lxml')
    clean_review = clean_review.get_text().replace('"', "")
    clean_review = clean_review.replace('\n', ' ').replace('\r', '').replace('\t', '')
    clean_review = clean_review.replace('\s{2,}', ' ')
    return clean_review


def load_data(file_path, year_range=[2008, 2017], usefulCount_range=[0, 10000]):
    # Read CSV
    df = pd.read_csv(file_path, encoding='utf-8')

    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['review', 'condition', 'date', 'rating', 'usefulCount'])

    # Filter the reviews by the input year_range
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[(df.date.dt.year >= year_range[0]) & (df.date.dt.year <= year_range[1]), :]

    # Get most common conditions (by review count)
    top_conditions = list(df.groupby('condition').count().reset_index().sort_values(by='uniqueID', ascending=False)[:10]['condition'])
    df = df.loc[df['condition'].isin(top_conditions)]

    # Clean review text
    df['cleanReview'] = df['review'].apply(clean_reviews)

    # Create standardized usefulScoreLog column (log of usefulCount normalized to be between 0 and 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['usefulScoreLog'] = np.log(df['usefulCount']) / np.max(np.log(df['usefulCount']))
    df['usefulScoreLog'] = df['usefulScoreLog'].replace(-np.Inf, 0)

    # Cap the usefulCount to create a new target variable column
    df['usefulCountCapped'] = df['usefulCount'].apply(lambda row : cap_col_val(row, usefulCount_range))

    # Normalize usefulCountCapped
    df['usefulCountCappedNormalized'] = df['usefulCountCapped'] / max(df['usefulCountCapped'])

    # Create normalized rating (0 to 1) to be used as a metadata feature
    df['ratingNormalized'] = df['rating'] / np.max(df['rating'])

    # Cast the date column to be a date datatype and compute the review age (with 0 corresponding to the most recent review)
    df['daysOld'] = (max(df['date']) - df['date']).astype('timedelta64[s]') / (60*60*24)

    # Compute an age score as the log of daysOld normalized to be between 0 and 1
    with np.errstate(divide='ignore', invalid='ignore'):
        df['ageScore'] = df['daysOld'] / np.max(df['daysOld'])
    df['ageScore'] = df['ageScore'].replace(-np.Inf, 0)

    # Create onehot encoding for the condition so we can use it as a feature as well
    df = pd.concat([df, pd.get_dummies(df['condition'])], axis=1)


    # Split data into train and val
    train = df.sample(frac=0.75, random_state=8)
    val = df.loc[~df['uniqueID'].isin(train['uniqueID'])]

    return train, val



def cap_col_val(val, usefulCount_range):
    if val > usefulCount_range[-1]:
        val = usefulCount_range[-1]
    return val