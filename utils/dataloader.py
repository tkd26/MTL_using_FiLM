# -*- coding: utf-8 -*-
# パッケージのimport
import os.path as osp
from PIL import Image
import scipy.io
import numpy as np
import torch
import torch.utils.data as data

from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(rootpath, Hariharan=True):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    
    if Hariharan:
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'img', '%s.jpg')
        annopath_template = osp.join(rootpath, 'cls', '%s.mat')
        
        # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
        train_id_names = osp.join(rootpath + 'train.txt')
        val_id_names = osp.join(rootpath + 'val.txt')
    else:
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
        annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

        # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
        train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
        val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform, Hariharan=True, anno_class=21, cutout_id=None):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.Hariharan = Hariharan
        self.anno_class = anno_class
        self.cutout_id = cutout_id

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        if self.Hariharan:
            anno_class_mat = scipy.io.loadmat(anno_file_path) # matデータの読み込み
            anno_class_img = anno_class_mat['GTcls'][0][0][1]   # matデータからインデックスカラー情報を取り出す(numpy)
            anno_class_img = Image.fromarray(anno_class_img.astype(np.uint8)).convert('P') # numpy → PIL
        else:
            anno_file_path = self.anno_list[index]
            anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


def make_dataloaders(args, rootpath, 
                     color_mean=(0.485, 0.456, 0.406), color_std=(0.229, 0.224, 0.225)):
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=args.imgsize, color_mean=color_mean, color_std=color_std), anno_class=args.classes)

    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=args.imgsize, color_mean=color_mean, color_std=color_std), anno_class=args.classes)

    # DataLoader作成
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader