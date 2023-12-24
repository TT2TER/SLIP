from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_images_from_folder(folder_path, image_size, device):
    images = []
    filenames = []
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            raw_image = Image.open(image_path).convert('RGB')
            image = transform(raw_image).unsqueeze(0).to(device)
            images.append(image)
            filenames.append(filename)

    return images,filenames

def process_batch(images, filenames, model, device, subset,batch_size=64):
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)

        batch_images = torch.cat(images[start_idx:end_idx], dim=0)  # Concatenate the image tensors

        with torch.no_grad():
            captions = model.generate(batch_images, sample=False, num_beams=3, max_length=20, min_length=5) 
            # If using nucleus sampling, uncomment the line below and comment the line above
            # captions = model.generate(batch_images, sample=True, top_p=0.9, max_length=20, min_length=5) 

        for idx, caption in enumerate(captions):
            #输出图片的文件名和生成的caption
            print(filenames[start_idx + idx], caption)
            filename_no_ext, _ = os.path.splitext(filenames[start_idx + idx])
            #输出每个caption为txt，存到./output_dataset/captions/{subset}/{filename}.txt
            #新建文件夹
            if not os.path.exists(f'./output_dataset/captions'):
                os.makedirs(f'./output_dataset/captions')
            if not os.path.exists(f'./output_dataset/captions/{subset}'):
                os.makedirs(f'./output_dataset/captions/{subset}')
            with open(f'./output_dataset/captions/{subset}/{filename_no_ext}.txt', 'w') as f:
                f.write(caption)




image_size = 384

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base',prompt='cartoon drawing of')
model.eval()
model = model.to(device)

if __name__ == "__main__":
    # Define the output folder paths
    image_output_folder = './output_dataset/image'
    text_output_folder = './output_dataset/text'

    # Process images and captions
    batch_size = 1
    image_paths = []
    for subset in [ 'valid','train', 'test']:
        subset_folder = os.path.join(image_output_folder, subset)
        images, filenames = load_images_from_folder(subset_folder, image_size=image_size, device=device)
        process_batch(images, filenames,model, device, subset,batch_size=batch_size)
