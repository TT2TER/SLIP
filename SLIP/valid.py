import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *


def valid(model: nn.Module, 
          valid_data_loader, 
          loss_fun,
          tokenizer = init_tokenizer(),
          device=torch.device("cuda:0")):
    model.eval()
    with torch.no_grad():
        for image, text in tqdm(valid_data_loader, desc="Validating..."):
            tokenize = tokenizer(text, padding='max_length', truncation=True, max_length=21, return_tensors="pt")
            
            tokens, labels  = tokenize["input_ids"][:,:-1], tokenize["input_ids"][:,1:]
            image, tokens, labels = image.to(device), tokens.to(device), labels.to(device)

            output = model(image, tokens)
            loss = loss_fun(output.view(-1, output.size(-1)), labels.view(-1))
    return loss.item()
