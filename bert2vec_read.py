#!

import pdb
import os
from bert_serving.client import BertClient
import traceback
import time
import arctic
from arctic import Arctic
import quandl
import pymongo

def init():
    '''
    运行在本地，60没有外部端口
    '''
    c = pymongo.MongoClient(host='localhost', port=27018)
    db = c.bert2vec
    coll = db.guizhou
    store = Arctic('localhost')
    library = store['bert2vec']
    return coll,library

def get_all_from_guizhou(limit=10):
    allsamp = coll.find({})[:limit]
    return allsamp

def get_all_arr(coll,library):
    allsample = get_all_from_guizhou()
    for line in allsample:
        print(line)
        pdb.set_trace()
        sentvec = line['sentvec']
        sent = line['sent']
        my_array = library.read(sentvec).data
        meta = library.read(sentvec).metadata
        yield my_array, meta,sent

if __name__ == '__main__':
    coll, library = init()
    gen = get_all_arr(coll, library)
    print(gen.__next__())


