import config
import torch.nn as nn
import transformers

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1) ## 1 coz itsa binary classification

    def forward(self, ids, mask, token_type_ids): ## these args are inpt to BertModel
        _, o2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        """out 1(aka last hidden state) is seq of hidden states for each and every token for all batches.for example, if u have 512 token (MAX_LEN) so u have
512 vectors of size 768 for each sample in batch. o2 is only the frst one (cls token of bert). o2 is pooler opt from bert pooler layer.. We can also perform max,
avg pooling on out1 which is why i place "_" in its place."""

        bo = self.bert_drop(o2)
        output = self.out(b0)
        return output



































































    
