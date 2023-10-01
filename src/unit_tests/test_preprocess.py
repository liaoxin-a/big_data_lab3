import configparser
import os
import unittest
import pandas as pd
import sys

import torch
sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from data import get_train_val_split
from predict import load_test_data
config = configparser.ConfigParser()
config_path=r'config\config.ini'
config.read(config_path)



class TestDataMaker(unittest.TestCase):

    def test_test_data(self):
        ids_path=r'dataset\test_ids.csv'
        features_path=r'dataset\test_features.npy'
        self.assertEqual(type(load_test_data(ids_path,features_path)), torch.utils.data.dataloader.DataLoader)
    
    def test_split_data(self):
        items=list(range(10))
        train_ids, val_ids=get_train_val_split(items,config['DEFAULT'].getint('num_folds'))
        self.assertEqual(len(train_ids)/len(val_ids),config['DEFAULT'].getint('num_folds')-1)



if __name__ == "__main__":
    unittest.main()
