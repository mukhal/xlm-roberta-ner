from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn


class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path,n_labels, hidden_size=768):
        super().__init__()

        self.n_labels = n_labels
        self.classification_head = nn.Linear(hidden_size, n_labels)
        self.xlmr = XLMRModel.from_pretrained(pretrained_path)

    def forward(self, inputs_ids, labels, labels_mask, bpe_valid_ids):
        '''
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask
        Computes a forward pass through the sequence tagging model.

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.xlmr.forward(inputs_ids, features_only=True)
        bsz, max_seq_len, hidden_size = transformer_out.size()

        sequence_output = self.dropout(valid_output)
        logits = self.classification_head(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if labels_mask is not None:
                active_loss = labels_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def encode_string(self, s):
        """
        takes a string and returns a list of token ids
        """

        tensor_ids = self.xlmr.task.source_dictionary.encode_line(s, append_eos=False,
                                                                  add_if_not_exist=False)

        return tensor_ids.cpu().numpy().tolist()
