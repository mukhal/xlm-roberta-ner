from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn

class XLMRForTokenClassification(nn.Module):


    def __init__(self, config):
        super().__init__()

        self.n_labels = config.n_labels
        self.classification_head = nn.Linear(config.hidden_size, config.n_labels)
        self.xlmr = XLMRModel.from_pretrained(config.pretrained_path)

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



        
        valid_output = torch.zeros(bsz, max_seq_len, hidden_size, dtype=torch.float32,device='cuda')
        
        for i in range(bsz):
            jj = -1
            for j in range(max_seq_len):
                    if bpe_valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = transformer_out[i][j]

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



