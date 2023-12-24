from datasets import load_dataset
import os
from PIL import Image
import io

# 加载Parquet数据集
print("start")
dataset = load_dataset("parquet", data_files={
    #需要先从https://huggingface.co/datasets/jmhessel/newyorker_caption_contest/tree/main/explanation下载对应的数据
    'test': './datasets/test-00000-of-00001.parquet',
    'train': './datasets/train-00000-of-00001.parquet',
    'valid': './datasets/validation-00000-of-00001.parquet'
})

# 定义保存图像的主文件夹路径
image_output_folder = './output_dataset/image'
text_output_folder = './output_dataset/text'

# 创建保存图像的文件夹
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(text_output_folder, exist_ok=True)

# 遍历数据集中的每个样本
for idx, sample in enumerate(dataset['valid']):
    # 获取比赛编号
    contest_number = sample['contest_number']

    # 创建保存图像的子文件夹路径
    contest_folder = os.path.join(image_output_folder, str('valid'))
    os.makedirs(contest_folder, exist_ok=True)

    # 图像文件名，格式为：valid_{索引值}.jpg
    image_filename = f"valid_{idx}.jpg"

    # 图像完整路径
    image_path = os.path.join(contest_folder, image_filename)

    # 将图像转换为bytes并保存
    with open(image_path, 'wb') as image_file:
        # If the image is in the 'image' field
        if isinstance(sample['image'], bytes):
            image_file.write(sample['image'])
        # If the image is a Pillow Image object
        elif isinstance(sample['image'], Image.Image):
            sample['image'].save(image_file, format='JPEG')
        else:
            # Handle other cases accordingly
            pass
    #获取'image_description'
    image_description = sample['image_description']
    #创建保存文本的子文件夹路径
    contest_folder = os.path.join(text_output_folder, str('valid'))
    os.makedirs(contest_folder, exist_ok=True)
    #文本文件名，格式为：valid_{索引值}.txt
    text_filename = f"valid_{idx}.txt"
    #文本完整路径
    text_path = os.path.join(contest_folder, text_filename)
    #将文本保存
    with open(text_path, 'w') as text_file:
        text_file.write(image_description)
    #打印进度
    if idx % 100 == 0:
        print(f"Processed {idx} samples")

for idx, sample in enumerate(dataset['train']):
    # 获取比赛编号
    contest_number = sample['contest_number']

    # 创建保存图像的子文件夹路径
    contest_folder = os.path.join(image_output_folder, str('train'))
    os.makedirs(contest_folder, exist_ok=True)

    # 图像文件名，格式为：train_{索引值}.jpg
    image_filename = f"train_{idx}.jpg"

    # 图像完整路径
    image_path = os.path.join(contest_folder, image_filename)

    # 将图像转换为bytes并保存
    with open(image_path, 'wb') as image_file:
        # If the image is in the 'image' field
        if isinstance(sample['image'], bytes):
            image_file.write(sample['image'])
        # If the image is a Pillow Image object
        elif isinstance(sample['image'], Image.Image):
            sample['image'].save(image_file, format='JPEG')
        else:
            # Handle other cases accordingly
            pass
    #获取'image_description'
    image_description = sample['image_description']
    #创建保存文本的子文件夹路径
    contest_folder = os.path.join(text_output_folder, str('train'))
    os.makedirs(contest_folder, exist_ok=True)
    #文本文件名，格式为：train_{索引值}.txt
    text_filename = f"train_{idx}.txt"
    #文本完整路径
    text_path = os.path.join(contest_folder, text_filename)
    #将文本保存
    with open(text_path, 'w') as text_file:
        text_file.write(image_description)
    #打印进度
    if idx % 100 == 0:
        print(f"Processed {idx} samples")

for idx, sample in enumerate(dataset['test']):
    # 获取比赛编号
    contest_number = sample['contest_number']

    # 创建保存图像的子文件夹路径
    contest_folder = os.path.join(image_output_folder, str('test'))
    os.makedirs(contest_folder, exist_ok=True)

    # 图像文件名，格式为：test_{索引值}.jpg
    image_filename = f"test_{idx}.jpg"

    # 图像完整路径
    image_path = os.path.join(contest_folder, image_filename)

    # 将图像转换为bytes并保存
    with open(image_path, 'wb') as image_file:
        # If the image is in the 'image' field
        if isinstance(sample['image'], bytes):
            image_file.write(sample['image'])
        # If the image is a Pillow Image object
        elif isinstance(sample['image'], Image.Image):
            sample['image'].save(image_file, format='JPEG')
        else:
            # Handle other cases accordingly
            pass
    #获取'image_description'
    image_description = sample['image_description']
    #创建保存文本的子文件夹路径
    contest_folder = os.path.join(text_output_folder, str('test'))
    os.makedirs(contest_folder, exist_ok=True)
    #文本文件名，格式为：test_{索引值}.txt
    text_filename = f"test_{idx}.txt"
    #文本完整路径
    text_path = os.path.join(contest_folder, text_filename)
    #将文本保存
    with open(text_path, 'w') as text_file:
        text_file.write(image_description)
    #打印进度
    if idx % 100 == 0:
        print(f"Processed {idx} samples")

print("end")