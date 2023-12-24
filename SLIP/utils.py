from transformers import BertTokenizer


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer