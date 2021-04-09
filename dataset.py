import config
import torch

class BERTDataset:
    def __init__(self, review, target): ## imported config coz also need tkenizer and max leng which is presend there
        
        self.review = review ## list of reviews
        self.target = target ## list of nums(0 or 1)
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item): ## takes item and retruns items from dataset like ids , token ids
        review = str(self.review[item]) ##just a sanity check
        review = "".join(review.split()) ##removing weird spaces

        ##now encode it..encode_plus can encode 2 strs at a time, but we have 1 only so we set 2nd to none and then additional params of the func
        
        inputs =  self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True,
            max_length = self.max_len
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        
        ##in BERT we pad on right side
        ids = ids + ([0] * padding_length)
        mask  = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids' : torch.tensor(ids, dtype = torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
            'targets' : torch.tensor(self.target[item], dtype = torch.float)
            }
    ## Float coz 1 linear layer for opt, use long fo 2 opts it also depends on the loss fucntion u use. cross_entropy uses
    ## NOW DATA IS READY, GOOTO ENGINE.PY AND COMPLETE IT










        
        
