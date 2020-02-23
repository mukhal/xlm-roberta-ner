# XLM-Roberta NER
Fine-tuning of the [XLM-Roberta](https://arxiv.org/abs/1911.02116) cross-lingual architecture for Sequence Tagging, namely Named Entity Recognition. 

The code is inspired by [BERT-NER](https://github.com/kamalkraj/BERT-NER) repo by kamalkraj.


## Requirements 
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
The code expects the data directory passed to contain 3 dataset splits: `train.txt`, `valid.txt` and `test.txt`.

```
python main.py 
      --data_dir=data/coNLL-2003/  \
      --task_name=ner   \
      --output_dir=model_dir/   \
      --max_seq_length=16   \
      --num_train_epochs 1  \
      --do_eval \
      --warmup_proportion=0.1 \
      --pretrained_path pretrained_models/xlmr.$PARAM_SET/ \
      --learning_rate 0.00007 \
      --do_train \
      --eval_on test \
      --train_batch_size 4
```

By default, the best model on the validation set is saved to `args.output_dir`. This model is then loaded and tested on the test set, if `--do_eval` and `--eval_on test`.

## Results

### CoNLL-2003
The below models were trained with the following settings:

```
--max_seq_length=128
--num_train_epochs 10
--warmup_proportion=0.0 
--learning_rate 3e-5  
--gradient_accumulation_steps 4 
--dropout 0.2 
--train_batch_size 32
```

| Model | Dev F1 | Test F1  |
|---|---|---|
| XLMR-Base |    |   |
| XLMR-Large  | 95.29  |  91.14 |
| XLMR-Large (Frozen) | | |

