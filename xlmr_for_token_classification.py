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
            inputs_ids: tensor of size (bsz, max_seq_len)
            labels: tensor of size (bsz, max_seq_len)
            labels_mask
        Computes a forward pass through the sequence tagging model.
        
        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.xlmr.forward(inputs_ids, features_only=True)
        bsz, max_seq_len, hidden_size = transformer_out.size()

        masked_transformer_out = torch.zeros(bsz, max_seq_len, hidden_size)

        
        logits = self.classification_head(transformer_out)  # (batch, max_seq_len, n_labels)



