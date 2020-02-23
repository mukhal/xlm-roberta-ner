from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import torch
import torch.nn.functional as F


def add_xlmr_args(parser):
     """
     Adds training and validation arguments to the passed parser
     """

     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--pretrained_path", default=None, type=str, required=True,
                         help="pretrained XLM-Roberta model path")
     parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train.")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     # Other parameters
     parser.add_argument("--cache_dir",
                         default="",
                         type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--do_train",
                         action='store_true',
                         help="Whether to run training.")
     parser.add_argument("--do_eval",
                         action='store_true',
                         help="Whether to run eval or not.")
     parser.add_argument("--eval_on",
                         default="dev",
                         help="Whether to run eval on the dev set or test set.")
     parser.add_argument("--do_lower_case",
                         action='store_true',
                         help="Set this flag if you are using an uncased model.")
     parser.add_argument("--train_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--eval_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for eval.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=3,
                         type=int,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     parser.add_argument('--gradient_accumulation_steps',
                         type=int,
                         default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument('--fp16',
                         action='store_true',
                         help="Whether to use 16-bit float precision instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument('--dropout', 
                         type=float, default=0.3,
                         help = "training dropout probability")
     
     parser.add_argument('--freeze_model', 
                         action='store_true', default=False,
                         help = "whether to freeze the XLM-R base model and train only the classification heads")
                                   
     return parser


def evaluate_model(model, eval_dataset, label_list, batch_size, device):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list, 1)}

     for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          valid_ids = valid_ids.to(device)
          l_mask = l_mask.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)

          logits = torch.argmax(logits, dim=2)
          logits = logits.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()

          for i, cur_label in enumerate(label_ids):
               temp_1 = []
               temp_2 = []

               for j, m in enumerate(cur_label):
                    if valid_ids[i][j]:  # if it's a valid label
                         temp_1.append(label_map[m])
                         temp_2.append(label_map[logits[i][j]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='Macro')

     return f1, report
