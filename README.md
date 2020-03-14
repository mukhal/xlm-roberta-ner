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
git clone https://github.com/mohammadKhalifa/xlm-roberta-ner.git
cd xlm-roberta-ner/
mkdir pretrained_models 
wget -P pretrained_models https://dl.fbaipublicfiles.com/fairseq/models/xlmr.$PARAM_SET.tar.gz
tar xzvf pretrained_models/xlmr.$PARAM_SET.tar.gz  --directory pretrained_models/
rm -r pretrained_models/xlmr.$PARAM_SET.tar.gz
```

## Training and evaluating
The code expects the data directory passed to contain 3 dataset splits: `train.txt`, `valid.txt` and `test.txt`.

Training arguments : 

```
 -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --pretrained_path PRETRAINED_PATH
                        pretrained XLM-Roberta model path
  --task_name TASK_NAME
                        The name of the task to train.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval or not.
  --eval_on EVAL_ON     Whether to run eval on the dev set or test set.
  --do_lower_case       Set this flag if you are using an uncased model.
  --train_batch_size TRAIN_BATCH_SIZE
                        Total batch size for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Total batch size for eval.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of training to perform linear learning rate
                        warmup for. E.g., 0.1 = 10% of training.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --no_cuda             Whether not to use CUDA when available
  --seed SEED           random seed for initialization
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --fp16                Whether to use 16-bit float precision instead of
                        32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --loss_scale LOSS_SCALE
                        Loss scaling to improve fp16 numeric stability. Only
                        used when fp16 set to True. 0 (default value): dynamic
                        loss scaling. Positive power of 2: static loss scaling
                        value.
  --dropout DROPOUT     training dropout probability
  --freeze_model        whether to freeze the XLM-R base model and train only
                        the classification heads
```
For example: 

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
      -- dropout 0.2
```

If you want to use the XLM-R model's outputs as features without finetuning, Use the `--freeze_model` argument.

By default, the best model on the validation set is saved to `args.output_dir`. This model is then loaded and tested on the test set, if `--do_eval` and `--eval_on test`.

## Results
### CoNLL-2003
I tried to reproduce the results in the paper by training the models using the following settings:

```
--max_seq_length=128
--num_train_epochs 10
--warmup_proportion=0.0 
--learning_rate 6e-5  
--gradient_accumulation_steps 4 
--dropout 0.2 
--train_batch_size 32
```
I got the following F1 scores:

| Model | Dev F1 | Test F1  |
|---|---|---|
| XLMR-Base |   95.29 | 91.14  |
| XLMR-Large  | 96.14  |  91.81 |

*The above results are close to those reported in the paper but a bit worse, probably due to the difference in experimental settings.*

