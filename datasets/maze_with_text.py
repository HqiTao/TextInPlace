import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import datasets.dataset_utils as dataset_utils


class MazeTextDataset(data.Dataset):
    def __init__(self, args, database_folder="database", queries_folder="queries", 
                 floor="all", positive_dist_threshold=10):
        self.dataset_name = args.dataset_name
        self.dataset_folder = os.path.join(args.datasets_folder, self.dataset_name, "images")
        self.database_folder = self.dataset_folder + "/test/" + database_folder
        self.queries_folder = self.dataset_folder + "/test/" + queries_folder
        self.database_paths = dataset_utils.read_images_paths(self.database_folder, get_abs_path=True)
        self.queries_paths = dataset_utils.read_images_paths(self.queries_folder, get_abs_path=True)
        self.floor = floor

        if self.floor == "all":
            pass
        elif isinstance(self.floor, int) and self.floor <= 5: 
            new_paths = []
            for path in self.queries_paths:
                if int(path.split("@")[3]) == self.floor:
                    new_paths.append(path)
            self.queries_paths = new_paths

            new_paths = []
            for path in self.database_paths:
                if int(path.split("@")[3]) == self.floor:
                    new_paths.append(path)
            self.database_paths = new_paths

        resize_test_imgs = args.resize_test_imgs
        image_size = args.resize
        
        #### Read paths and UTM coordinates for all images.
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([
            (float(path.split("@")[1]) + 50 * (int(path.split("@")[3]) - 1),
            float(path.split("@")[2]))
            for path in self.database_paths])
        self.queries_utms = np.array([
            (float(path.split("@")[1]) + 50 * (int(path.split("@")[3]) - 1),
            float(path.split("@")[2]))
            for path in self.queries_paths])

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=positive_dist_threshold, return_distance=False
        )

        self.images_paths = self.database_paths + self.queries_paths

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        transforms_list = []
        if resize_test_imgs:
            # Resize to image_size along the shorter side while maintaining aspect ratio
            transforms_list += [transforms.Resize(image_size, antialias=True)]
        transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.base_transform = transforms.Compose(transforms_list)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img =Image.open(image_path).convert("RGB")
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        if self.floor == "all":
            return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
        elif isinstance(self.floor, int):
            return f"< {self.dataset_name}_{self.floor}F - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query
