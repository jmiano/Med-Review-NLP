import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import numpy as np


def clean_reviews(review):
    clean_review = BeautifulSoup(review, features='lxml')
    return clean_review.get_text().replace('"', "")


def load_data(file_path):
    # Read CSV
    df = pd.read_csv(file_path, encoding='utf-8')

    # Get most common conditions (by review count)
    top_conditions = list(df.groupby('condition').count().reset_index().sort_values(by='uniqueID', ascending=False)[:10]['condition'])
    df = df.loc[df['condition'].isin(top_conditions)]

    # Clean review text
    df['cleanReview'] = df['review'].apply(clean_reviews)

    # Create standardized usefulScore column (log of usefulCount normalized to be between 0 and 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['usefulScore'] = np.log(df['usefulCount']) / np.max(np.log(df['usefulCount']))
    df['usefulScore'] = df['usefulScore'].replace(-np.Inf, 0)

    # Split data into train and val
    train = df.sample(frac=0.75, random_state=8)
    val = df.loc[~df['uniqueID'].isin(train['uniqueID'])]

    return train, val