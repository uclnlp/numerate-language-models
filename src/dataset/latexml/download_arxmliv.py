from __future__ import print_function
import urllib2
import re
from bs4 import BeautifulSoup
import os
import time
import random


class Timer(object):
    def __init__(self, mode):
        self.mode = mode
        self.n_called = 0
        self.total_time = 0.0
        self.avg_num_requests_per_minute = 30
        self.reset_every = 5*60  # in secs
        self.reset_delay = 10  # in secs

    def get_delay(self):
        self.n_called += 1
        if self.mode == 'unlimited':
            interval = 0.0
        elif self.mode == 'careful':
            interval = random.expovariate(self.avg_num_requests_per_minute/60.0)
            self.total_time += interval
            if self.total_time > self.reset_every:
                self.total_time = 0
                interval = self.reset_delay
        return interval


def get_page_urls(retval='no_problems'):
    base_url = 'http://arxmliv.kwarc.info/retval_detail.php'
    url = '{base_url}?retval={retval}'.format(base_url=base_url, retval=retval)
    req = urllib2.Request(url)
    page_urls = []
    try:
        response = urllib2.urlopen(req)
        html = response.read()
        soup = BeautifulSoup(html, "lxml")
        for a_link in soup.find('center').find_all('a'):
            href = a_link['href']
            page_urls.append(base_url + href)
    except urllib2.URLError as e:
        print(e.reason)
    page_urls = page_urls[1:]
    return page_urls


def project_base_url_gen(retval='no_problems', skip_pages=0):
    page_urls = get_page_urls(retval=retval)
    if skip_pages:
        page_urls = page_urls[skip_pages:]
    for page_num, page_url in enumerate(page_urls, start=skip_pages):
        print('Page {page_num} ({page_url})'.format(page_num=page_num, page_url=page_url))
        req = urllib2.Request(page_url)
        try:
            response = urllib2.urlopen(req)
            html = response.read()
            soup = BeautifulSoup(html, "lxml")
            for table in soup.find_all('table',  {'border': '1'}):
                for a_link in table.find_all('a'):
                    href = a_link['href']
                    yield 'http://arxmliv.kwarc.info/' + href
        except urllib2.URLError as e:
            print(e.reason)


def download_file(url, proj_dir, xml_file_name):
    print(url)
    # get xml file
    req = urllib2.Request(url)
    try:
        response = urllib2.urlopen(req)
        xml = response.read()
        soup = BeautifulSoup(xml, "lxml")
        xml = soup.prettify().encode('utf-8')
        if len(xml.split('\n')) < 10:
            print('Skip: Too short...')
            return False
    except urllib2.URLError as e:
        print('Skip:', e.reason)
        return False
    # write file
    os.makedirs(proj_dir)
    output_path = os.path.join(proj_dir, xml_file_name)
    with open(output_path, 'wb') as fout:
        fout.write(xml)
    return True


def main(output_dir='../../../data/arxmliv'):
    return  # CAUTION with re-running!
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timer = Timer('careful')
    n_files = 0
    for base_url in project_base_url_gen(retval='no_problems', skip_pages=0):
        m = re.search(r'/files/(\d+)/(\d+\.\d+)/', base_url)
        if m:
            proj_dir, proj_id = m.group(1), m.group(2)
            #xml_file = '{proj_id}.tex.xml'.format(proj_dir=proj_dir, proj_id=proj_id)
            xml_file_name = '{proj_id}.noparse.xml'.format(proj_dir=proj_dir, proj_id=proj_id)
            xml_url = '{base_url}/{xml_file}'.format(base_url=base_url, xml_file=xml_file_name)
            # make directory and path to file
            proj_dir = os.path.join(output_dir, proj_dir, proj_id)
            success = download_file(xml_url, proj_dir, xml_file_name)
            if success:
                n_files += 1
                delay = timer.get_delay()
                #print('Delay: ', delay)
                time.sleep(delay)
            print('Downloaded: ', n_files)
        #if n_files > 1000: break

if __name__ == '__main__':
    main()
