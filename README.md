# BERT-for-sentiment-prediction
Used BERT model with fine tuning it with one additional output layer for the sentiment prediction. This model takes some sequence as an input, it then looks left and right several times (perform computation based on conditional probabilities) and returns the sentiment of the input sentence. Training was done for 5 epochs to achieve the accuracy of 84%. The model is also served using python’s flask API.

