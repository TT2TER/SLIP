import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.EncoderDecoderModel import EncoderDecoder
from data.data_manager import manage_data2
from data.data_loader import load_data
from predict import predict
from evaluate import evaluate
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
            
        torch.save(model.state_dict(), "model/checkpoints_finetune.pth")
        valid_loss = valid(model, valid_data_loader=valid_data_loader, 
                           loss_fun=criterion, tokenizer=tokenizer, device=device)
        
        print("Epoch: {}, Train Loss, Test Loss: {}".format(epoch, loss.item(), valid_loss))


if __name__=="__main__":
    BATCH_SIZE = 64

    train_dataset = manage_data2(root="data/dataset/output_dataset", max_words=100, dataset="train")
    valid_dataset = manage_data2(root="data/dataset/output_dataset", max_words=100, dataset="val")
    # train_data_loader, valid_data_loader = load_data(train_dataset, valid_dataset, batch_size=BATCH_SIZE)

    model = EncoderDecoder(encoder_model=args.model)
    print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))                          
    if args.load == 'yes':
        model.load_state_dict(torch.load("model/checkpoints_finetune.pth"))

    train_data_loader, valid_data_loader = load_data(train_dataset, valid_dataset, batch_size=BATCH_SIZE)
    train(model, train_data_loader, valid_data_loader, max_tokens_len=51,
        epochs=2, lr=0.0001, weight_decay=0.001, loss_reduction="mean", device=torch.device("cuda:0"))

    path = "data/dataset/output_dataset/image/valid/valid_2.jpg"
    steps = 20
    max_len = 50
    prompt = ""
    device=torch.device("cuda:0")
    evaluate(model, valid_dataset, device=device, method="greedy")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="greedy")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="sample")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="temperature_sample", temperature = 0.25)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="topk_sample", k=10)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="beam_search", beam_width=10)
    plt.imshow(plt.imread(path))
    plt.show()

    model.load_state_dict(torch.load("model/checkpoints.pth"))
    evaluate(model, valid_dataset, device=device, method="greedy")