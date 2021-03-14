# 爬虫代码，爬图片的，输入list的id，就可以全部爬下来
import os
import sys
import socket
import socks

PROXY_IP = '127.0.0.1'
PROXY_PORT = 9050
default_socket = socket.socket
socks.set_default_proxy(socks.SOCKS5, PROXY_IP, PROXY_PORT)
socket.socket = socks.socksocket  # 将socket.socket这个类变成了别的类，所以实现了代理的目的

import requests
from html.parser import HTMLParser
from PIL import Image
from io import BytesIO
import pandas as pd
import glob

print ('variable initialized')
err_num = 0 
err_list = []
url_list = []

class ImageHTMLParser(HTMLParser):
    def __init__(self):
        super(ImageHTMLParser, self).__init__()
        self.cnt = -1
        self.start = False
        self.image_url = ""

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if (tag == 'td' and d.get('id', '') == 'img_container'):
            print ('[Parser] found right tr', tag, attrs)
            self.start = True
            self.cnt   = -1
            
        if (self.start and tag == 'img'):
            self.cnt += 1
            if (self.cnt == 1) : 
                print ('[Parser] found right img: ' + d['src'])
                self.image_url = 'https:' + d['src']
        pass
    def handle_endtag(self, tag):
        if (tag == 'td'):
            self.start = False
        pass
    def handle_data(self, data):
        pass

    def get_img_url(self) : 
        return self.image_url

if sys.version_info[0] != 3 : 
    print ("this script requires python3")
    exit()

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}

def get_img_by_url(id, url) :
    id = str(id)
    img_req = requests.get(url, headers=headers)
    f = BytesIO(img_req.content)
    im = Image.open(f)
    #import pdb
    #pdb.set_trace()
    im.save('./img/{id}.jpg'.format(id=id))

def get_img_by_id(id):
    global err_num, err_list
    id_str = str(id)
    r = requests.get('https://www.dpchallenge.com/image.php?IMAGE_ID=' + id_str, headers=headers)
    #requests.warnings
    img_parser = ImageHTMLParser()
    img_parser.feed(r.text)
    url = img_parser.get_img_url()
    if url == '' : 
        print ('[ERROR] img {id} can not be found'.format(id=id))
        err_num += 1
        err_list.append(str(id))
        return 
    try : 
        get_img_by_url(id, url)
    except Exception as e: 
        print (e)
        print ('[ERROR] img {id} can not be get'.format(id=id))
        err_num += 1
        err_list.append(str(id))

import time
task_list = [] # task-list
if __name__ == "__main__":
    print ('[LOG] load task')
    task = pd.read_csv('./dataset.txt', header=None, names=['uid', 'iid', 'r', 'q'])['iid'].unique().tolist()
    
    task_mask = set(glob.glob('./img/*'))
    #import pdb
    #pdb.set_trace()
    print (len(task))
    print ('[LOG] start')

    for id in task:
        if './img/'+str(id)+'.jpg' not in task_mask: 
            get_img_by_id(id)
    #        time.sleep(3)
        else :                              print ('[LOG] skip {id}'.format(id=id))
    print ('[LOG] end successful')
    print ('[ERROR]:')
    print (err_num)
    print (err_list)
