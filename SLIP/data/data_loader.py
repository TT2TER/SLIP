from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

from .data_manager import manage_data

class COCODataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image_path, text_list = self.dataset[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        text = random.choice(text_list)
        return image, text

    def __len__(self):
        return len(self.dataset)
    

def load_data(train_dataset, valid_dataset, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    train_dataset = COCODataset(train_dataset, transform=transform)
    valid_dataset = COCODataset(valid_dataset, transform=transform)
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False)
    return train_data_loader, valid_data_loader
