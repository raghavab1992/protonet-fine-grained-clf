import os


PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/home/raghava_balusu_walmartlabs_com/Work/oneshotml/data/Grocery_products'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
