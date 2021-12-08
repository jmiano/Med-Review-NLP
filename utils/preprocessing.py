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


def load_data(file_path, year_range=[2008, 2017], usefulCount_range=[0, 10000], usefulCount_quantile=None,
              quantiles_for_class=[0.25, 0.5, 0.75]):
    # Read CSV
    df = pd.read_csv(file_path, encoding='utf-8')

    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['review', 'condition', 'date', 'rating', 'usefulCount'])

    # Get most common conditions (by review count)
    top_conditions = list(df.groupby('condition').count().reset_index().sort_values(by='uniqueID', ascending=False)[:10]['condition'])
    df = df.loc[df['condition'].isin(top_conditions)]

    # Filter the reviews by the input year_range
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[(df.date.dt.year >= year_range[0]) & (df.date.dt.year <= year_range[1]), :]

    # Create onehot encoding for the condition so we can use it as a feature as well
    df = pd.concat([df, pd.get_dummies(df['condition'])], axis=1)

    # Clean review text
    df['cleanReview'] = df['review'].apply(clean_reviews)

    # Create standardized usefulScoreLog column (log of usefulCount normalized to be between 0 and 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['usefulScoreLog'] = np.log(df['usefulCount']) / np.max(np.log(df['usefulCount']))
    df['usefulScoreLog'] = df['usefulScoreLog'].replace(-np.Inf, 0)

    # Cap the usefulCount to create a new target variable column
    if usefulCount_quantile is not None:
        usefulCount_range = [0, int(df['usefulCount'].quantile(q=usefulCount_quantile))]
    df['usefulCountCapped'] = df['usefulCount'].apply(lambda row : cap_col_val(row, usefulCount_range))

    # Normalize usefulCountCapped
    df['usefulCountCappedNormalized'] = df['usefulCountCapped'] / max(df['usefulCountCapped'])

    # Create normalized rating (0 to 1) to be used as a metadata feature
    df['ratingNormalized'] = df['rating'] / np.max(df['rating'])

    # Cast the date column to be a date datatype and compute the review age (with 0 corresponding to the most recent review)
    df['daysOld'] = (max(df['date']) - df['date']).astype('timedelta64[s]') / (60*60*24)

    # Compute an age score as daysOld normalized to be between 0 and 1
    with np.errstate(divide='ignore', invalid='ignore'):
        df['ageScore'] = df['daysOld'] / np.max(df['daysOld'])
    df['ageScore'] = df['ageScore'].replace(-np.Inf, 0)

    # Create a usefulCountClass column to treat usefulness prediction as a classification problem
    if quantiles_for_class is not None:
        buckets = get_buckets(df=df, quantiles=quantiles_for_class)
        df['usefulCountClass'] = df['usefulCount'].apply(lambda row : assign_bucket(row, buckets))

    # Split data into train and val
    train = df.sample(frac=0.75, random_state=8)
    val = df.loc[~df['uniqueID'].isin(train['uniqueID'])]

    return train, val



def cap_col_val(val, usefulCount_range):
    if val > usefulCount_range[-1]:
        val = usefulCount_range[-1]
    return val


def assign_bucket(val, buckets):
    for i, bucket in enumerate(buckets):
        if bucket[0] <= val < bucket[1]:
            new_val = i
    return new_val


def get_buckets(df, quantiles):
    cutoffs = []
    buckets = []
    for i in quantiles:
        cutoffs.append(df['usefulCount'].quantile(q=i))
    for i in range(1, len(cutoffs)):
        if i == 1:
            buckets.append([0, cutoffs[i-1]])
        buckets.append([cutoffs[i-1], cutoffs[i]])
        if i == len(cutoffs) - 1:
            buckets.append([cutoffs[i], np.inf])
    if len(cutoffs)==1:
        buckets.append([0, cutoffs[0]])
        buckets.append([cutoffs[0], np.inf])
    return buckets