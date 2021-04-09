import config
import torch
import flask
from flask import Flask
from flask import request
from model import BERTBaseUncased

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"

def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review =  str(sentence)
    review = "".join(review.split()) ##removing weird spaces

        ##now encode it..encode_plus can encode 2 strs at a time, but we have 1 only so we set 2nd to none and then additional params of the func
        
    inputs =  tokenizer.encode_plus(
        review,
        None,
        add_special_tokens = True,
        max_len = max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ##in BERT we pad on right side
    ids = ids + ([0] * padding_length)
    mask  = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    
    ids =  torch.tensor(ids, dtype = torch.long).unsqueeze(0) #coz dataloader return batcher so need to add 1 more dim
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype = torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype = torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype = torch.long)
    mask = mask.to(DEVICE, dtype = torch.long)
    

    
    outputs = model(
        ids = ids,
        mask = mask,
        token_type_ids = token_type_ids
    )
    

    outputs = torch.sigmoid(outputs)
    return outputs[0][0].cpu().detach().numpy() #coz opt is 2d but there is only 1 val in opt




#createing end point
@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    positive_prediction = sentence_prediction(sentence, model = MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "postivie" : str(positive_prediction),
        "negative" : str(negative_prediction),
        "sentence" : str(sentence)
    }
    return flask.jsonify(response)



if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.to(DEVICE)
    MODEL.eval()    
    app.run()
