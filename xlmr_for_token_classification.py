from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn


class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size=768, dropout_p=0.1, label_ignore_idx=0):
        super().__init__()

        self.n_labels = n_labels
        self.classification_head = nn.Linear(hidden_size, n_labels)
        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
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
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        #transformer_out = self.dropout(transformer_out)

        bsz, max_seq_len, hidden_size = transformer_out.size()
        valid_output = torch.zeros(bsz, max_seq_len, hidden_size)

        for i in range(bsz):
            for j in range(max_seq_len):
                if valid_mask[i][j]:
                    valid_output[i][j] = transformer_out[i][j]

        valid_output = self.dropout(valid_output)
        logits = self.classification_head(valid_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
