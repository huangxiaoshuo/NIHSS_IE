import torch.nn as nn
import sys
sys.path.append('..')
from main.model.layers import crf
from main.model.layers import normalization
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertModel

class BERTLSTMCRF(BertPreTrainedModel):
    def __init__(self, config, label2id, device, num_layers=2, lstm_dropout=0.35):
        super(BERTLSTMCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(label2id))
        self.init_weights()
        self.bilstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2,
                             batch_first=True,
                             num_layers=num_layers,
                             dropout=lstm_dropout,
                             bidirectional=True)

        self.layer_norm = normalization.LayerNorm(config.hidden_size)
        self.crf = crf.CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device, is_bert=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        seq_output = outputs[0] # Sequence of hidden-states at the output of the last layer of the model
        seq_output = self.dropout(seq_output)
        seq_output, _ = self.bilstm(seq_output)
        seq_output = self.layer_norm(seq_output)
        logits = self.classifier(seq_output)
        return logits

    def forward_loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        features = self.forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if labels is not None:
            return features, self.crf.calculate_loss(features, tag_list=labels, lengths=input_lens)
        else:
            return features, None