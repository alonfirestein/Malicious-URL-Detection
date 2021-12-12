import pandas as pd
import urllib
import string
from Helpers.preprocess.utils import *


def get_content_based_features(url):
    features = dict()
    urlparse = urllib.parse.urlparse(url)
    html = get_html(url)
    pq = get_pq(html)
    suspicious_functions = ['escape','eval','link','unescape','exec','search']
#     valid_tags =
    if not html or not pq:
        features['is_active'] = False
        return pd.Series(features)
    scripts = pq('script')
    features['is_active'] = True
    features['url_page_entropy'] = get_entropy(html)
    features['number_of_script_tags'] = len(scripts)
    features['script_to_body_ratio'] = len(scripts.text())/len(html)
    features['number_of_page_tokens'] = len(html.lower().split())
    features['number_of_sentences'] = len(html.split('.'))
    features['number_of_punctuations'] = len([i for i in html if i in string.punctuation and i not in ['<', '>', '/']])
    features['number_of_distinct_tokens'] = len(set([i.strip() for i in html.lower().split()]))
    features['number_of_capitalizations'] = len([i for i in html if i.isupper()])
    features['number_iframes'] = len(pq('iframe') + pq('frame'))
    features['number_objects'] = len(pq('object'))
    features['number_embeds'] = len(pq('embed'))
    features['number_hyperlinks'] = len(pq('a'))
    features['number_of_whitespace'] = len([i for i in html if i == ' '])
    features['number_of_hidden_tags'] = len(pq('.hidden')+pq('#hidden')+pq('*[visibility="none"]')+pq('*[display="none"]'))
    features['number_of_html_tags'] = len(pq('*'))
    features['number_of_double_documents'] = len(pq('html') + pq('body') + pq('title')) - 3
    for suspicious_func in suspicious_functions:
        features[f'number_of_suspicious_function_{suspicious_func}'] = sum([suspicious_func in script.text().lower() for script in scripts.items()])
    features['average_script_length'] = average_script_length(pq)
#     features['number_of_suspicious_elements'] = len([i for i in [i.tag for i in pq('*')] if i not in valid_tags])#TODO: valid_tags
    features['average_number_of_tokens_in_sentence'] = average_number_of_tokens_in_sentence(html)
    return pd.Series(features)
