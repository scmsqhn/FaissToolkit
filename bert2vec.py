import os

from bert_serving.client import BertClient
import traceback
import time
from arctic import Arctic
import pymongo
# import pdb
import logging
from logging.handlers import RotatingFileHandler
import re
from mylogger import logger
# =================================
# logger setting
# ================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#  logger = logging.getLogger(__name__)
logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")

handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


rHandler = RotatingFileHandler("log.txt", maxBytes=1 * 1024, backupCount=3)
rHandler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rHandler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(rHandler)
logger.addHandler(console)


def init():
    # Connect to Local MONGODB
    logger.info('init start')
    store = Arctic('localhost')
    # Create the library - defaults to VersionStore
    store.initialize_library('bert2vec')
    # Access the library
    library = store['bert2vec']
    c = pymongo.MongoClient(host='localhost', port=27018)
    db = c.bert2vec
    coll = db.guizhou
    # lib,conn,db,coll
    logger.info('init ok')
    return library, c, db, coll


def init_bert():
    logger.info('init bert')
    bc = BertClient(
        ip='58.17.133.80',
        show_server_config=True,
        timeout=10000,
        port=18087,
        port_out=15005)
    logger.info('init bert SUCC')
    return bc


def insert_sent_into_mongo(bc, library, c, db, coll, PATH,sent_or_chars='chars'):
    PATH = '/data/guizhou_19042/glue_dir/NCA/'
    for _, _, filenames in os.walk(PATH):
        for filename in filenames:
            if r'\,' in filename:
                if not filename.split(r'\.')[1] in ['tsv', 'csv', 'txt']:
                    # if not filename in ['train.tsv','test.csv','eval.tsv']:
                    continue
        try:
            open(os.path.join(PATH, filename), 'r').read()
            logger.info('handler the file %s' % filename)
        except BaseException:
            traceback.print_exc()
            continue
        for line in open(os.path.join(PATH, filename), 'r').readlines():
            try:
                parts = re.split('[\t\r\n]', line)
                for part in parts:
                    if len(re.findall('[\u4e00-\u9fa5]', part)) > 0:
                        print("RIGHT " + line)
                        if sent_or_chars == 'sent':
                            part = [part]
                        else:
                            part = list(part)
                        sentvec = bc.encode(part)
                        time_ = str(int(time.time())) + str(hash(part))
                        sentvecid = time_
                        library.write(
                            sentvecid, sentvec, metadata={
                                'sentvecid': sentvecid})
                        json_cell = {
                            'id': time_,
                            'sent': part,
                            'sentvec': sentvecid,
                        }
                        print(coll.insert(json_cell))
                        logger.info('handler the sent %s' % part)
                        # pdb.set_trace()
            except BaseException:
                traceback.print_exc()
                print("WRONG " + line)


if __name__ == '__main__':
    PATH = '.'
    lib, c, db, coll = init()
    bc = init_bert()
    insert_sent_into_mongo(bc, lib, c, db, coll, PATH)

# item = library.read('AAPL')
# aapl = item.data
# metadata = item.metadata
