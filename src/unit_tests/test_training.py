import configparser
import os
import unittest
import sys
import torch
sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from model import ClassifierModel

config = configparser.ConfigParser()
config_path=r'config\config.ini'
config.read(config_path)
FEATURES_DIM=config['DEFAULT'].getint('features_dim')
BATCH_SIZE=config['DEFAULT'].getint('batch_size')
NUM_CLASSES=config['DEFAULT'].getint('num_classes')

class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.classifier_model = ClassifierModel()
        

    def test_forward(self):
        self.classifier_model.train()
        input=torch.rand((BATCH_SIZE,5,FEATURES_DIM))#b,5,feature_dim
        self.assertEqual(self.classifier_model(input).shape, (BATCH_SIZE,NUM_CLASSES))



if __name__ == "__main__":
    unittest.main()

