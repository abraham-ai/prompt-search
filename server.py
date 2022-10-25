import argparse
import datetime
import requests
import os
import random
import numpy as np
from PIL import Image

from eden.block import Block
from eden.hosting import host_block
from eden.datatypes import Image as EdenImage
from eden.webhook import WebHook

from clip_search import find_top_k_matches

eden_block = Block()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-workers', help='maximum number of workers to be run in parallel', required=False, default=1, type=int)
parser.add_argument('-p', '--port', help='localhost port', required=False, type=int, default=5656)
parser.add_argument('-rh', '--redis-host', help='redis host', required=False, type=str, default='0.0.0.0')
parser.add_argument('-rp', '--redis-port', help='redis port', required=False, type=int, default=6379)
parser.add_argument('-l', '--logfile', help='filename of log file', required=False, type=str, default=None)
args = parser.parse_args()

my_args = {
    "top_k": 5,
    "mode": "img2img2txt",
    #"image": EdenImage(),
    "image_url": ""
}

@eden_block.run(args=my_args)
def run(config):
    input_img = Image.open(requests.get(config["image_url"], stream=True).raw)
    matches, similarities = find_top_k_matches(input_img, config["mode"], config["top_k"])
    similarities = [float(s) for s in similarities]
    results = {"matches": matches, "similarity": similarities}
    return results  


host_block(
    block = eden_block,
    port = args.port,
    host = "0.0.0.0",
    max_num_workers = args.num_workers,
    redis_port = args.redis_port,
    redis_host = args.redis_host,
    logfile = args.logfile, 
    log_level = 'debug',
    requires_gpu = False
)
