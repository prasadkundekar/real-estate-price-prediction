import pandas as pd
import numpy as np


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
    df = df.dropna()

    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

    def convert_sqft(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df[df['total_sqft'].notnull()]

    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    df['location'] = df['location'].apply(lambda x: x.strip())
    location_stats = df['location'].value_counts()
    locations_less_than_10 = location_stats[location_stats <= 10]

    df['location'] = df['location'].apply(
        lambda x: 'other' if x in locations_less_than_10 else x
    )

    df = df[~(df['total_sqft'] / df['bhk'] < 300)]

    return df
