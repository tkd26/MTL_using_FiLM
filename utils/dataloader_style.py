# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
import torch.utils.data as data

class StyleDataset(data.Dataset):

    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform
        
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)
    
    def __getitem__(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]
        img = np.asarray(img)
        while len(img.shape)!=3:
            index += 1
            image_file_path = self.img_list[index+1]
            img = Image.open(image_file_path)
            img = np.asarray(img)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img
    
    
def make_style_dataloader(args, num, 
                     color_mean=(0.485, 0.456, 0.406), color_std=(0.229, 0.224, 0.225)):
    
    style_img_list = glob.glob("/export/data/dataset/WikiArt/artist/pablo-picasso/image/*.jpg")
    #style_img_list = glob.glob("/export/data/dataset/WikiArt/artist/claude-monet/image/*.jpg")
    
    while len(style_img_list) <= num:
        style_img_list += style_img_list
    style_img_list = style_img_list[:num]    
    style_dataset = StyleDataset(style_img_list, 
                               transform=transforms.Compose([
                                   transforms.Resize((args.imgsize,args.imgsize)),
                                   transforms.ToTensor(), 
                                   transforms.Normalize(color_mean, color_std),
                                   ])
                                )

    # DataLoader作成
    style_dataloader = data.DataLoader(
        style_dataset, batch_size=args.batch_size, shuffle=True)
    
    return style_dataloader