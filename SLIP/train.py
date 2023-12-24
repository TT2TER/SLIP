import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer

from model.EncoderDecoderModel import EncoderDecoder
from data.data_manager import manage_data
from data.data_loader import load_data
from predict import predict
from valid import valid
from utils import *
import argparse

parser = argparse.ArgumentParser(description='caption parameters')
parser.add_argument('--model', type=str, default='vit_b_16', help="model")
parser.add_argument('--load', type=str, default='yes', help="load params")
args = parser.parse_args()

def train(model: nn.Module, 
          train_data_loader, 
          valid_data_loader, 
          epochs=5, 
          max_tokens_len = 16,
          lr=0.001, 
          weight_decay=0.0001, 
          loss_reduction="mean",
          tokenizer = init_tokenizer(),
          device=torch.device("cuda:0")):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction=loss_reduction, label_smoothing=0.1)
    for epoch in range(epochs):
        model.train()
        total_loss = []
        tqdm_iterator = tqdm(train_data_loader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}')
        for image, text in tqdm_iterator:
            # tokenize
            tokenize = tokenizer(text, padding='max_length', truncation=True, max_length=max_tokens_len, return_tensors="pt")
            tokens, labels  = tokenize["input_ids"][:,:-1], tokenize["input_ids"][:,1:]
            image, tokens, labels = image.to(device), tokens.to(device), labels.to(device)

            output= model(image, tokens)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss))
            
        torch.save(model.state_dict(), "model/checkpoints.pth")
        valid_loss = valid(model, valid_data_loader=valid_data_loader, 
                           loss_fun=criterion, tokenizer=tokenizer, device=device)
        
        print("Epoch: {}, Train Loss, Test Loss: {}".format(epoch, loss.item(), valid_loss))


if __name__=="__main__":
    BATCH_SIZE = 64

    train_dataset = manage_data(root="data/dataset/COCO", max_words=50, dataset="train", version="2017")
    valid_dataset = manage_data(root="data/dataset/COCO", max_words=50, dataset="val", version="2017")
    # train_data_loader, valid_data_loader = load_data(train_dataset, valid_dataset, batch_size=BATCH_SIZE)

    model = EncoderDecoder(encoder_model=args.model)
    print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))                          
    if args.load == 'yes':
        model.load_state_dict(torch.load("model/checkpoints.pth"))
    for _ in range(1):
        train_data_loader, valid_data_loader = load_data(train_dataset, valid_dataset, batch_size=BATCH_SIZE)
        train(model, train_data_loader, valid_data_loader, max_tokens_len=16,
          epochs=1, lr=0.0001, weight_decay=0.00001, loss_reduction="mean", device=torch.device("cuda:0"))

    predict(model, "data/dataset/COCO/val2017/000000000139.jpg", steps=20, max_len=50, 
            prompt="a picture of ", device=torch.device("cuda:0"), method="greedy")