import re
import math
import socket
import requests
from pyquery import PyQuery


def url_host_is_ip(urlparse):
    host = urlparse.netloc
    pattern = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    match = pattern.match(host)
    return match is not None


def get_ip(urlparse):
    try:
        ip = urlparse.netloc if url_host_is_ip(urlparse) else socket.gethostbyname(urlparse.netloc)
        return ip
    except:
        return None


def url_has_port_in_string(urlparse):
    has_port = urlparse.netloc.split(':')
    return len(has_port) > 1 and has_port[-1].isdigit()


def get_entropy(text):
    text = text.lower()
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum([p * math.log(p) / math.log(2.0) for p in probs])


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
