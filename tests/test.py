import sys
sys.path.append('..')

from xlmr_for_token_classification import XLMRForTokenClassification
from collections import namedtuple
import unittest, logging 

from data_utils import *

logging.basicConfig(level=logging.DEBUG)

class TestTextEncoding(unittest.TestCase):

    def setUp(self):
        Config = namedtuple('config', ['pretrained_path', 'n_labels', 'hidden_size'])
        config = Config('../pretrained_models/xlmr.base', 5, 768)

        #self.model = XLMRForTokenClassification(config)

    
    def test_loading_data(self):

        processor = NerProcessor()
        train_examples = processor.get_train_examples('testing_data')

        features = convert_examples_to_features(train_examples, processor.get_labels(), 
        128, self.get_dummy_encoding_method())

    def get_dummy_encoding_method(self):
        return lambda s:[100] * len(s)

if __name__ =='__main__':

    unittest.main()