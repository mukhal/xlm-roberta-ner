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
export PARAM_SET=base # change to large to use the large architecture

# clone the repo
git clone https://github.com/mohammadKhalifa/XLMR-NER.git
cd XLMR-NER/
mkdir pretrained_models 
wget -P pretrained_models https://dl.fbaipublicfiles.com/fairseq/models/xlmr.$PARAM_SET.tar.gz
tar xzvf pretrained_models/xlmr.$PARAM_SET.tar.gz  --directory pretrained_models/

```

## Training and evaluating
The code expects the data directory passed to contain 3 dataset splits: `train.txt`, `valid.txt` and `test.txt`. The code directory 

```

python main.py --data_dir=data/conll2003/ \
  --task_name=ner  \
  --output_dir=model_dir/ \
  --max_seq_length=16 \
  --num_train_epochs 1 \
  --do_eval \
  --warmup_proportion=0.1 \
  --pretrained_path pretrained_models/xlmr.$PARAM_SET/ \
  --learning_rate 0.00007 \
  --do_train \
  --eval_on test \
  --train_batch_size 4

```


