import os
from tqdm import tqdm
import json

captions_path = r"E:\datasets\COCO\annotations\captions_train2017.json"
txt_path = r"E:\datasets\COCO\annotations\captions_train2017_txt"

# 读取json文件
with open(captions_path, 'r') as f1:
    dictionary = json.load(f1)

# 得到images和annotations信息
images_value = dictionary.get("images")
annotations_value = dictionary.get("annotations")

# 使用字典按图像id分组annotations
annotations_by_image_id = {}
for annotation in annotations_value:
    image_id = annotation.get("image_id")
    if image_id not in annotations_by_image_id:
        annotations_by_image_id[image_id] = []
    annotations_by_image_id[image_id].append(annotation.get("caption"))

# 遍历图像id列表
for image_info in tqdm(images_value):
    image_id = image_info.get("id")
    img_name = image_info.get("file_name").split(".")[0]
    file_name = os.path.join(txt_path, img_name + '.txt')

    # 写入描述信息到文本文件
    if image_id in annotations_by_image_id:
        captions = annotations_by_image_id[image_id]
        with open(file_name, 'w') as f2:
            f2.write("\n".join(captions))

print('Over!')
