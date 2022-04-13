import os
from easydict import EasyDict

CONF = EasyDict()

CONF.ROOT = "/Users/venus/AI_lift/multi_section"
CONF.DATA_DIR = os.path.join(CONF.ROOT, "data")
CONF.SRC = os.path.join(CONF.ROOT, "src")