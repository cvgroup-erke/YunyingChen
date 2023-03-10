from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

import numpy as np
from PIL import Image
import random
import os
from config import load_config


def collate_fn(batch):
    # use the customized collate_fn to filter out bad inputs
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class BBCDataProcess(Dataset):
    def __init__(self, mode, file_list):
        self.input_transform = T.Compose([T.ToPILImage(),
                                          T.Resize(size=(256, 256)),
                                          T.ToTensor(),
                                          T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToPILImage(),
                                           T.Resize(size=(256, 256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
        self.file_list = file_list
        self.mode = mode
        if mode == 'basic':
            pass
        else:
            self.input_transform_256 = T.Compose([T.ToPILImage(),
                                                  T.Resize(size=(256, 256), interpolation=Image.LANCZOS),
                                                  T.Grayscale(),
                                                  T.ToTensor(),
                                                  T.Normalize(0.5, 0.5)
                                                  ])
            self.input_transform_128 = T.Compose([T.ToPILImage(),
                                                  T.Resize(size=(128, 128), interpolation=Image.LANCZOS),
                                                  T.Grayscale(),
                                                  T.ToTensor(),
                                                  T.Normalize(0.5, 0.5)
                                                  ])
            self.input_transform_64 = T.Compose([T.ToPILImage(),
                                                 T.Resize(size=(64, 64), interpolation=Image.LANCZOS),
                                                 T.Grayscale(),
                                                 T.ToTensor(),
                                                 T.Normalize(0.5, 0.5)
                                                 ])

    def transform(self, image, img_type, size):
        # crop
        image_t = image[:530, 140:]
        if img_type == 'source':
            if size == 256:
                return self.input_transform(image_t)
            elif size == 128:
                return self.input_transform_128(image_t)
            elif size == 64:
                return self.input_transform_64(image_t)
            else:   # size = 0
                return self.input_transform_256(image_t)
        elif img_type == 'target':
            return self.target_transform(image_t)

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.file_list)

    def __getitem__(self, index: int):  # -> Tuple(torch.Tensor, torch.Tensor):
        # Return the input tensor and output tensor for training
        if self.mode =='flownet':
            image_names = self.file_list[index].split(' ')
            source_image_name = image_names[0]
            target_image_name = image_names[1][:-1]
            dir_path = os.path.dirname(source_image_name)
            # img_name = self.file_list[index]
            source_image = np.asarray(Image.open(source_image_name))
            target_image = np.asarray(Image.open(target_image_name))
            
            #   t1/t2
            # dir_path = os.path.dirname(source_image)
            img_name = source_image_name.split('/')[-1]
            img_name = int(img_name.split('.jpg')[0])
            img_t1_s = os.path.join(dir_path,str(img_name+1)+'.jpg')
            img_t2_s = os.path.join(dir_path,str(img_name+1)+'.jpg')
            img_t1_t = img_t1_s.replace('source','target')
            img_t2_t = img_t2_s.replace('source','target')

            img_t1_s = np.asarray(Image.open(img_t1_s))
            img_t1_t = np.asarray(Image.open(img_t1_t))

            img_t2_s = np.asarray(Image.open(img_t2_s))
            img_t2_t = np.asarray(Image.open(img_t2_t))
            
            try:
                input_image, input_image_256, input_image_128, input_image_64, target_image = \
                        self.transform(source_image, 'source', 256), self.transform(source_image, 'source', 0), \
                        self.transform(source_image, 'source', 128), self.transform(source_image, 'source', 64), \
                        self.transform(target_image, 'target', 256)

                #t1
                input_imaget1, input_imaget1_256, input_imaget1_128, input_imaget1_64, target_imaget1 = \
                    self.transform(img_t1_s, 'source', 256), self.transform(img_t1_s, 'source', 0), \
                    self.transform(img_t1_s, 'source', 128), self.transform(img_t1_s, 'source', 64), \
                    self.transform(img_t1_t, 'target', 256)

                #t2
                input_imaget2, input_imaget2_256, input_imaget2_128, input_imaget2_64, target_imaget2 = \
                    self.transform(img_t2_s, 'source', 256), self.transform(img_t2_s, 'source', 0), \
                    self.transform(img_t2_s, 'source', 128), self.transform(img_t2_s, 'source', 64), \
                    self.transform(img_t2_t, 'target', 256)


            except:
                # there are some strange exceptions
                return None
        else:
            image_names = self.file_list[index].split(' ')
            source_image_name = image_names[0]
            target_image_name = image_names[1][:-1]
            # img_name = self.file_list[index]
            source_image = np.asarray(Image.open(source_image_name))
            target_image = np.asarray(Image.open(target_image_name))
            try:
                if self.mode == 'basic':
                    input_image, target_image = self.transform(source_image, 'source', 256), \
                                                self.transform(target_image, 'target', 256)
                else:
                    input_image, input_image_256, input_image_128, input_image_64, target_image = \
                        self.transform(source_image, 'source', 256), self.transform(source_image, 'source', 0), \
                        self.transform(source_image, 'source', 128), self.transform(source_image, 'source', 64), \
                        self.transform(target_image, 'target', 256)
            except:
                # there are some strange exceptions
                return None

        if self.mode == 'basic':
            return input_image, target_image
            
        elif self.mode == 'flownet':
            input_images = [input_image,input_imaget1,input_imaget2]
            input_images_256 = [input_image_256,input_imaget1_256,input_imaget2_256]
            input_images_128  =[input_image_128,input_imaget1_128,input_imaget2_128]
            input_images_64 = [input_image_64,input_imaget1_64,input_imaget2_64] 
            target_images = [target_image,target_imaget1,target_imaget2]     
            return input_images, input_images_256, input_images_128, input_images_64, target_images

        else:
            return input_image, input_image_256, input_image_128, input_image_64, 

class BBCDataset:
    def __init__(self, args):
        random.seed(args.seed)

        # generate the list contains all data/images
        data_list_file = 'data_list2.txt'
        with open(data_list_file) as f:
            all_img_ids = f.readlines()
        random.shuffle(all_img_ids)

        # generate training & validating image lists
        if args.val_ratio == 0:
            # Finally use all image to train
            self.train_dataset = BBCDataProcess(args.mode, all_img_ids)
            self.val_dataset = BBCDataProcess(args.mode, random.sample(all_img_ids, int(0.25 * len(all_img_ids))))
        else:
            train_size = int(len(all_img_ids) * (1 - args.val_ratio))
            train_ids = all_img_ids[:train_size]
            val_ids = all_img_ids[train_size:]

            # generate training & validating
            self.train_dataset = BBCDataProcess(args.mode, train_ids)
            self.val_dataset = BBCDataProcess(args.mode, val_ids)


if __name__ == '__main__':
    args = load_config()
    bbc_dataset = BBCDataset(args)
