from xlmr_for_token_classification import XLMRForTokenClassification
from collections import namedtuple

Config = namedtuple('config', ['pretrained_path', 'n_labels', 'hidden_size'])
config = Config('pretrained_models/xlmr.base', 5, 768)

model = XLMRForTokenClassification(config)