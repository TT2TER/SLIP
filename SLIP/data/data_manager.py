import os
from tqdm import tqdm
import re


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )     
    return caption.rstrip('\n').strip(' ') 


def manage_data(root="E:\datasets\COCO", max_words=50, dataset = "train", version="2017"):
    # 遍历文件夹下所有文件名，存在list中
    if dataset == "train":
        if version == "2017":
            image_root = os.path.join(root, "train2017")
        elif version == "2014":
            image_root = os.path.join(root, "train2014")  # Add this line for handling version "2014"
    elif dataset == "val":
        if version == "2017":
            image_root = os.path.join(root, "val2017")
        elif version == "2014":
            image_root = os.path.join(root, "val2014")  # Add this line for handling version "2014"
    image_id_list = []
    for filename in os.listdir(image_root):
        # 拼接文件路径
        image_id_list.append(filename.split(".")[0])
    if dataset == "train":
        if version == "2017":
            image_text_root = os.path.join(root, "annotations/captions_train2017_txt")
        elif version == "2014":
            image_text_root = os.path.join(root, "annotations/captions_train2014_txt")
    elif dataset == "val":
        if version == "2017":
            image_text_root = os.path.join(root, "annotations/captions_val2017_txt")
        elif version == "2014":
            image_text_root = os.path.join(root, "annotations/captions_val2014_txt")
    id2text = {}
    for filename in tqdm(os.listdir(image_text_root), desc="Loading data..."):
        # 拼接文件路径
        with open(os.path.join(image_text_root, filename), "r") as f:
            text_list = f.readlines()
            id2text[filename.split(".")[0]] = [pre_caption(line, max_words) for line in text_list]

    # image_text_pair = [image_path, [text_id1, text_id2, ...]]
    image_text_pair = []    
    for id in image_id_list:
        image_text_pair.append([os.path.join(image_root, id + ".jpg"), id2text[id]])

    # 写入文件txt
    # with open(os.path.join(root, "train.txt"), "w") as f:
    #     for pair in image_text_pair:
    #         f.write(pair[0] + "\n")
    #         for text in pair[1]:
    #             f.write(text + "\n")
    #         f.write("\n")
    print("Dataset len: ", len(image_text_pair))
    return image_text_pair


def manage_data2(root="data/dataset/output_dataset", max_words=50, dataset = "train"):
    # 遍历文件夹下所有文件名，存在list中
    if dataset == "train":
        image_root = os.path.join(root, "image")
        image_root = os.path.join(image_root, "train")
    elif dataset == "val":
        image_root = os.path.join(root, "image")
        image_root = os.path.join(image_root, "valid")
    image_id_list = []
    for filename in os.listdir(image_root):
        # 拼接文件路径
        image_id_list.append(filename.split(".")[0])
    if dataset == "train":
        image_text_root = os.path.join(root, "text")
        image_text_root = os.path.join(image_text_root, "train")
    elif dataset == "val":
        image_text_root = os.path.join(root, "text")
        image_text_root = os.path.join(image_text_root, "valid")
    id2text = {}
    for filename in tqdm(os.listdir(image_text_root), desc="Loading data..."):
        # 拼接文件路径
        with open(os.path.join(image_text_root, filename), "r") as f:
            text_list = f.readlines()
            id2text[filename.split(".")[0]] = [pre_caption(line, max_words) for line in text_list]

    # image_text_pair = [image_path, [text_id1, text_id2, ...]]
    image_text_pair = []    
    for id in image_id_list:
        image_text_pair.append([os.path.join(image_root, id + ".jpg"), id2text[id]])

    # 写入文件txt
    # with open(os.path.join(root, "train.txt"), "w") as f:
    #     for pair in image_text_pair:
    #         f.write(pair[0] + "\n")
    #         for text in pair[1]:
    #             f.write(text + "\n")
    #         f.write("\n")
    print("Dataset len: ", len(image_text_pair))
    return image_text_pair