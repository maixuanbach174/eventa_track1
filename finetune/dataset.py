from torch.utils.data import Dataset, DataLoader
import csv
import os
from PIL import Image
from torchvision import transforms

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class ImageCaptionDataset(Dataset):
    def __init__(self, annotations_path, image_dir,transform=None):
        with open(annotations_path, 'r') as f:
            reader = csv.reader(f)
            self.annotations = list(reader)[1:]
            self.image_paths = [os.path.join(image_dir, f"{row[0]}.jpg") for row in self.annotations]
        self.image_dir = image_dir
        if transform is None:
            self.transform = get_default_transform()
    
        

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        return image_tensor, annotation[1]
    
if __name__ == "__main__":
    dataset = ImageCaptionDataset(annotations_path="dataset/gt_train.csv", image_dir="dataset/train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    images, captions, image_ids = next(iter(dataloader))
    print(images[0].shape)
    print(captions[0])
    print(image_ids[0])

