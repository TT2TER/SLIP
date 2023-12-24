import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import transforms
from PIL import Image
from utils import *


def predict(model: nn.Module, 
            image_path, 
            steps = 20,
            max_len = 50, 
            tokenizer = init_tokenizer(),
            prompt = 'a picture of ',
            device=torch.device("cuda:0"),
            method="greedy", 
            **kwargs):
    print("Method:", method)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        # 显示图片
        plt.imshow(image)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
        text = prompt.split(" ")
        if method == "beam_search":
            beam_width = kwargs["beam_width"]
            beam_list = [(text, 1)]
            for _ in range(steps):
                new_beam_list = []
                for text, score in beam_list:
                    tokenize = tokenizer(' '.join(text), return_tensors="pt")["input_ids"]
                    tokens = tokenize[:,:-1].to(device)
                    output = model(image, tokens)
                    value, slide = torch.topk(output[:,-1], beam_width)
                    for i in range(beam_width):
                        new_beam_list.append(
                            (text+[tokenizer.convert_ids_to_tokens(slide.squeeze(0)[i].item())], 
                             score*value.squeeze(0)[i].item()))
                beam_list = sorted(new_beam_list, key=lambda x: x[1], reverse=True)[:beam_width]
                if beam_list[0][0][-1] == "[SEP]":
                    break
                if len(beam_list[0][0])>max_len: break
            text = beam_list[0][0]

        else:
            for _ in range(steps):
                tokenize = tokenizer(' '.join(text), return_tensors="pt")["input_ids"]
                tokens = tokenize[:,:-1].to(device)
                output = model(image, tokens)
                if method == "greedy":
                    text.append(tokenizer.convert_ids_to_tokens(output.argmax(dim=-1)[:,-1].item()))
                elif method == "sample":
                    text.append(tokenizer.convert_ids_to_tokens(torch.multinomial(torch.softmax(output[:,-1], dim=-1), 1).item()))
                elif method == "temperature_sample":
                    temp = kwargs["temperature"]
                    text.append(tokenizer.convert_ids_to_tokens(torch.multinomial(torch.softmax(output[:,-1]/temp, dim=-1), 1).item()))
                elif method == "topk_sample":
                    k = kwargs["k"]
                    value, slide = torch.topk(output[:,-1], k)
                    text.append(tokenizer.convert_ids_to_tokens(slide.squeeze(0)[torch.multinomial(torch.softmax(value, dim=-1), 1).squeeze(0)].item()))
                else:
                    raise Exception("No such method")
                if text[-1] == "[SEP]":
                    break
                if len(text)>max_len: break
        predictions = tokenizer.convert_tokens_to_string([token for token in text if token not in special_tokens])
        print(predictions+".")


if __name__=="__main__":
    from model.EncoderDecoderModel import EncoderDecoder
    model = EncoderDecoder()
    model.load_state_dict(torch.load("model/checkpoints.pth"))
    path = "data/dataset/output_dataset/image/valid/valid_0.jpg"
    max_len = 50
    steps = 20
    prompt = "a picture of"
    device=torch.device("cuda:0")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="greedy")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="sample")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="temperature_sample", temperature = 0.2)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="topk_sample", k=5)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="beam_search", beam_width=10)
    plt.show()
    model.load_state_dict(torch.load("model/checkpoints.pth"))
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="greedy")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="sample")
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="temperature_sample", temperature = 0.2)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="topk_sample", k=5)
    predict(model, path, steps=steps, max_len=max_len, prompt=prompt, device=device, method="beam_search", beam_width=10)
    plt.show()