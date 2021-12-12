import pandas as pd
import urllib
from Helpers.preprocess.utils import *


def get_lexical_url_features(url):
    features = dict()
    try:
        urlparse = urllib.parse.urlparse(url)
        features["url_length"] = len(url)
        features["url_schema"] = urlparse.scheme
        features["url_path_length"] = len(urlparse.path)
        features["url_host_length"] = len(urlparse.netloc)
        features["url_host_is_ip"] = url_host_is_ip(urlparse)
        features['url_has_port_in_string'] = url_has_port_in_string(urlparse)
        features['number_of_digits'] = len([i for i in url if i.isdigit()])
        features['number_of_parameters'] = (0 if urlparse.query == '' else len(urlparse.query.split('&')))
        features['number_of_fragments'] = (len(urlparse.fragment.split('#')) - 1 if urlparse.fragment == '' else 0)
        features['is_encoded'] = '%' in url.lower()
        features['num_encoded_char'] = len([i for i in url if i == '%'])
        features['url_string_entropy'] = get_entropy(url)
        features['number_of_subdirectories'] = len(urlparse.path.split('/'))
        features['number_of_periods'] = len([i for i in url if i == '.'])
        speciel_words = ['client','admin','server','login']
        for word in speciel_words:
            features[f'has_{word}_in_string'] = (word in url.lower())
        features["tld"] = urlparse.netloc.split('.')[-1].split(':')[0]
    except:
        pass
    return pd.Series(features)
