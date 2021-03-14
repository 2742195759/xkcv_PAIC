import os
import json

if __name__ == '__main__':
    datapath = './data/AVA_PCap.json'
    outpaht  = './data/AVA_PCap_after_process.json'
    raw_data  = json.load(open(datapath))

