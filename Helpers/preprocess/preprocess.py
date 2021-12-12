import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import swifter
from Helpers.Constants import *
from Helpers.preprocess.host_based_features import get_host_based_features
from Helpers.preprocess.content_based_features import get_content_based_features
from Helpers.preprocess.lexical_url_features import get_lexical_url_features


def preprocess(data_path="Data/malicious_phish.csv",features_extractors=(get_lexical_url_features,)):
    df = pd.read_csv(data_path)
    df['type'] = pd.Categorical(df['type'])
    df['type'] = df['type'].cat.codes
    df['url'] = df['url'].swifter.apply(lambda url: (defualt_protocol + url) if (defualt_protocol not in url) else url)
    for features_extractor in features_extractors:
        features = df['url'].swifter.apply(features_extractor)
        df = pd.concat([df, features], axis=1)

    return df


def scale_data():
    pass
