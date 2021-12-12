import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import swifter
from Helpers.Constants import *
from Helpers.preprocess.host_based_features import get_host_based_features
from Helpers.preprocess.content_based_features import get_content_based_features
from Helpers.preprocess.lexical_url_features import get_lexical_url_features
import os


def preprocess(data_path="Data/malicious_phish.csv", features_extractors=(get_lexical_url_features,),
               cache=True, save_to_path="./cleaned.csv") -> pd.DataFrame:
    if cache and os.path.isfile(save_to_path):
        df = pd.read_csv(save_to_path, index_col=0)
        return df
    df = pd.read_csv(data_path)
    df['type'] = pd.Categorical(df['type'])
    df['type'] = df['type'].cat.codes
    df['url'] = df['url'].swifter.apply(lambda url: (defualt_protocol + url) if (defualt_protocol not in url) else url)
    for features_extractor in features_extractors:
        features = df['url'].swifter.apply(features_extractor)
        df = pd.concat([df, features], axis=1)
    df = df.replace({False: 0, True: 1})
    df = pd.concat([df, pd.get_dummies(df['url_schema']), pd.get_dummies(df['tld'])], axis=1)
    df = df.drop(['url_schema', 'tld'], axis=1)
    df.to_csv(save_to_path)
    return df


def scale_data():
    pass
