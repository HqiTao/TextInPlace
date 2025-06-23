import os
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="/root/autodl-tmp", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = os.path.join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(os.path.join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
        # self.images_paths = sorted(glob(os.path.join(dataset_folder_full_path, "MSLS*", "*.jpg"), recursive=True))
        self.transform = transforms.Compose([transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR) if args.resize_test_imgs else lambda x: x,
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    def __getitem__(self, index):
        return self.transform(Image.open(self.images_paths[index]).convert("RGB"))
    
    def __len__(self):
        return len(self.images_paths)
