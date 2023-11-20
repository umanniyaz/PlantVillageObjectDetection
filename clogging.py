from __future__ import absolute_import
import logging

import os

BASE_DIR = os.getcwd()
LOGGING_PATH = BASE_DIR + '/Logs'

if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)

def setup_logger(logger_name, log_file, mode='a', level=None):
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)
    else:
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(log_file, mode=mode)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        l.setLevel(level)
        l.addHandler(fileHandler)
        l.addHandler(streamHandler)

        return l