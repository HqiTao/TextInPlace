# https://github.com/amaralibey/gsv-cities

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

# GSV-Cities
TRAIN_CITIES = ['Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL',
                'Rome', 'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix',
                'TRT', 'Boston', 'Lisbon', 'Medellin', 'Minneapolis',
                'PRG', 'WashingtonDC', 'Brussels', 'London',
                'Melbourne', 'Osaka', 'PRS',]

INDOOR_DATASETS = ["StationB1", "StationB2", "Store1F", "Store4F", "StoreB1",]


class GSVCitiesDataset(Dataset):
    def __init__(self, args, cities=['London', 'Boston'], img_per_place=4, min_img_per_place=4): # 4 for SuperPlace
        super(GSVCitiesDataset, self).__init__()
        self.base_path = os.path.join(args.datasets_folder, "gsv_cities/")
        self.cities = cities
        self.is_inference = False

        assert img_per_place <= min_img_per_place, f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.transform = transforms.Compose([transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),])

        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()

        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')

    def __getitem__(self, index):

        if self.is_inference:
            place_id = self.places_ids[index]
            row = self.dataframe.loc[place_id].iloc[0]
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)
            img = self.transform(img)

            return img, torch.tensor([place_id])


        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        place = place.sample(n=self.img_per_place)
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            img = self.transform(img)

            imgs.append(img)

        # [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        # row.name is the index of the row, not to be confused with image name
        pl_id = row.name % 10**5
        pl_id = str(pl_id).zfill(7)

        value = row['panoid']
        if isinstance(value, float) and value.is_integer():
            panoid = str(int(value))  # 转换为整数后转字符串，避免使用 .rstrip(".0")
        else:
            panoid = str(value)
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
