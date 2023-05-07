import os
LOGGING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
TICKER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../','ticker')
if not os.path.exists(LOGGING_DIR):
	os.mkdir(LOGGING_DIR)
    
if not os.path.exists(TICKER_DIR):
	os.mkdir(TICKER_DIR)