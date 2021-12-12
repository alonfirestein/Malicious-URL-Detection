import pandas as pd
import urllib
from pyquery import PyQuery
import requests
from shodan import Shodan
from whois import whois
from waybackpy import Cdx
from datetime import datetime
from Helpers.preprocess.utils import *


def get_html(url):
    try:
        html = requests.get(url, timeout=5)
        html = html.text if html else None
    except:
        html = None
    return html


def get_pq(html):
    try:
        pq = PyQuery(html) if html else None
        return pq
    except:
        return None


def average_script_length(pq):
    scripts_lst = [len(script.text()) for script in pq("script").items()]
    scripts_len = len(scripts_lst)
    return (sum(scripts_lst) / scripts_len) if scripts_len != 0 else 0


def average_number_of_tokens_in_sentence(html):
    sen_lens = [len(i.split()) for i in html.split('.')]
    return sum(sen_lens) / len(sen_lens)


def number_of_included_elements(pq):
    toi = pq('script') + pq('iframe') + pq('frame') + pq('embed') + pq('form') + pq('object')
    toi = [tag.attr('src') for tag in toi.items()]
    return len([i for i in toi if i])


class HostFeatures:
    def __init__(self, url):
        try:
            self.url = url
            self.urlparse = urllib.parse.urlparse(self.url)
            self.host = get_ip(self.urlparse)
            self.now = datetime.now()
            self.whois = self.__get__whois_dict()
            self.shodan = self.__get_shodan_dict()
            self.snapshots = self.__get_site_snapshots()
            self.error = False
        except:
            self.error = True

    def __get__whois_dict(self):
        try:
            whois_dict = whois(self.host)
            return whois_dict
        except:
            return {}

    def __get_shodan_dict(self):
        api = Shodan('W6cy1PGcje0jJwKDBTgrqWSZioRpRmzg')
        try:
            host = api.host(self.host)
            return host
        except:
            return {}

    def __parse_whois_date(self, date_key):
        cdate = self.whois.get(date_key, None)
        if cdate:
            if isinstance(cdate, str) and 'before' in cdate:
                d = datetime.strptime('01-{}'.format(cdate.split()[-1]), '%d-%b-%Y')
            elif isinstance(cdate, list):
                d = cdate[0]
            else:
                d = cdate
        return d if cdate else cdate

    def __get_site_snapshots(self):
        try:
            snapshots = Cdx(self.urlparse.netloc).snapshots()
            snapshots = [snapshot.datetime_timestamp for snapshot in snapshots]
            return snapshots
        except:
            return []

    def number_of_subdomains(self):
        ln1 = self.whois.get('nets', None)
        ln2 = self.shodan.get('domains', None)
        ln = ln1 or ln2
        return len(ln) if ln else None

    def url_creation_date(self):
        return self.__parse_whois_date('creation_date')

    def url_expiration_date(self):
        return self.__parse_whois_date('expiration_date')

    def url_last_updated(self):
        return self.__parse_whois_date('updated_date')

    def url_age(self):
        try:
            days = (self.now - self.url_creation_date()).days
        except:
            days = None
        return days

    def url_intended_life_span(self):
        try:
            lifespan = (self.url_expiration_date() - self.url_creation_date()).days
        except:
            lifespan = None
        return lifespan

    def url_life_remaining(self):
        try:
            rem = (self.url_expiration_date() - self.now).days
        except:
            rem = None
        return rem

    def url_registrar(self):
        return self.whois.get('registrar', None)

    def url_registration_country(self):
        return self.whois.get('country', None)

    def url_host_country(self):
        return self.shodan.get('country_name', None)

    def url_num_open_ports(self):
        return len(self.shodan.get('ports', ''))

    def url_isp(self):
        return self.shodan.get('isp', '')

    def url_connection_speed(self):
        return requests.get('{}://{}'.format(self.urlparse.scheme, self.urlparse.netloc)).elapsed.total_seconds()

    def first_seen(self):
        try:
            return self.snapshots[0]
        except:
            return datetime.now()

    def get_os(self):
        oss = self.shodan.get('os', None)
        return oss

    def last_seen(self):
        try:
            ls = self.snapshots[-1]
            return ls
        except:
            return datetime.now()

    def days_since_last_seen(self):
        return (self.now - self.last_seen()).days

    def days_since_first_seen(self):
        return (self.now - self.first_seen()).days

    def average_update_frequency(self):
        snapshots = self.snapshots
        diffs = [(t - s).days for s, t in zip(snapshots, snapshots[1:])]
        l = len(diffs)
        if l > 0:
            return sum(diffs) / l
        else:
            return 0

    def number_of_updates(self):
        return len(self.snapshots)

    def ttl_from_registration(self):
        try:
            ttl_from_reg = (self.first_seen() - self.url_creation_date()).days
        except:
            ttl_from_reg = None
        return ttl_from_reg


def get_host_based_features(url):
    features = dict()
    features_extractor = HostFeatures(url)
    if features_extractor.error:
        return pd.Series()
    features['ttl_from_registration'] = features_extractor.ttl_from_registration()
    features['average_update_frequency'] = features_extractor.average_update_frequency()
    features['days_since_first_seen'] = features_extractor.days_since_last_seen()
    features['days_since_last_seen'] = features_extractor.days_since_last_seen()
    features['last_seen'] = features_extractor.last_seen()

    features['get_os'] = features_extractor.get_os()
    features['url_registrar'] = features_extractor.url_registrar()
    features['first_seen'] = features_extractor.first_seen()
    features['url_connection_speed'] = features_extractor.url_connection_speed()
    features['url_isp'] = features_extractor.url_isp()
    features['url_num_open_ports'] = features_extractor.url_num_open_ports()
    features['url_host_country'] = features_extractor.url_host_country()
    features['number_of_updates'] = features_extractor.number_of_updates()
    features['url_life_remaining'] = features_extractor.url_life_remaining()
    features['url_intended_life_span'] = features_extractor.url_intended_life_span()
    features['url_age'] = features_extractor.url_age()
    features['url_last_updated'] = features_extractor.url_last_updated()
    features['number_of_subdomains'] = features_extractor.number_of_subdomains()
    features['url_creation_date'] = features_extractor.url_creation_date()
    features['url_expiration_date'] = features_extractor.url_expiration_date()

    return pd.Series(features)
