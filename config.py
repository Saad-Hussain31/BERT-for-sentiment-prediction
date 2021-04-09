import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10 
BERT_PATH = "/home/saad/python_programming/bert/input"
MODEL_PATH = "/home/saad/python_programming/bert/input"
TRAINING_FILE = "/home/saad/python_programming/bert/input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case = True
    )
