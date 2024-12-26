import os 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, dataloader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class MulticlassSegmentation(Dataset):
    def __init__(self, image_dir, label_dir,color_to_class, transform=None,augmentation= False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.color_to_class = color_to_class
        self.transform = transform
        self.augmentation= augmentation

        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path= os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        image = TF.resize(image, (512, 512))
        label = TF.resize(label, (512, 512))
        
        

        # Maping RGB colors in labels to class indices
        label = np.array(label)
        mapped_label = np.zeros((label.shape[0], label.shape[1]), dtype = np.uint8)
        for color, class_index in self.color_to_class.items():
            mask = np.all(label == color, axis = -1)
            mapped_label[mask] = class_index
        label = torch.from_numpy(mapped_label)

        label = Image.fromarray(mapped_label)

        if self.augmentation:
            image, label = self._apply_augmentation(image, label)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor separately
        label = torch.tensor(np.array(label), dtype=torch.long)

        return image, label
    
    def _apply_augmentation(self, image, label):
            # Define augmentation transformations
            augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=30),
                
            ])

            # Apply augmentation to both image and label
            augmented = augmentation(image)
            label = augmentation(label)

            return augmented, label


        
# color_to_class = {
#     (155, 38, 182): 0,  # obstacles
#     (14, 135, 204): 1,  # water
#     (124, 252, 0): 2,   # nature
#     (255, 20, 147): 3,  # moving
#     (169, 169, 169): 4  # landable
# }

# transform = T.Compose([
#     T.Resize((512, 512)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# dataset = MulticlassSegmentation(
#     image_dir="/home/krish/Desktop/unet/classes_dataset/original_images",
#     label_dir="/home/krish/Desktop/unet/classes_dataset/label_images_semantic",
#     color_to_class=color_to_class,
#     transform=transform,
#     augmentation=True
# )


