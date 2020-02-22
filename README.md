# XLMR-NER
This repo implements finetuning the [XLM-Roberta](https://arxiv.org/abs/1911.02116) Cross-lingual architecture for Sequence Tagging, namely Named Entity Recognition. 

The code is inspired from the original BERT repo and the [BERT-NER](https://github.com/kamalkraj/BERT-NER) repo by kamalkraj.


## Requirments 
* `python 3.6+`
* `torch 1.x`
* `fairseq`
* `pytorch_transformers` (for AdamW and WarmpUpScheduler)


## Setting up

```bash
# clone the repo
git clone https://github.com/mohammadKhalifa/XLMR-NER.git

export PARAM_SET=base # change to large to use the large architecture

cd XLMR-NER/
mkdir pretrained_models 
wget -P pretrained_models https://dl.fbaipublicfiles.com/fairseq/models/xlmr.$PARAM_SET.tar.gz
tar xzvf pretrained_models/xlmr.$PARAM_SET.tar.gz  --directory pretrained_models/


```

