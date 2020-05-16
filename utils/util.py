import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torchvision
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from .data_augumentation import Normalize_Tensor
import os


def colorize_mask(mask):
    '''
    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    '''
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 100, 100, 100]

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def DeNormalize(tensors, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    tensors_denorm = []
    for tensor in tensors:
        tensor = transforms.functional.normalize(tensor, -1.0*mean/std, 1.0/std)
        tensors_denorm += [tensor.cpu().numpy()]
    return torch.Tensor(tensors_denorm)


def make_paintmask(imgs, n=1, size=30, mode='train'):
    w,h = imgs.shape[2:]
    masks = []
    imgs = imgs.cpu().numpy()
    for img in imgs:
        img = img.copy()
        mask = np.ones(img.transpose(1,2,0).shape)
        if mode=='eval': np.random.seed(1)
        x1_list = np.random.randint(0, w-1-size, n)
        if mode=='eval': np.random.seed(2)
        y1_list = np.random.randint(0, h-1-size, n)
        
        for i in range(n):
            x1 = x1_list[i]
            y1 = y1_list[i]
            x2 = x1 + size
            y2 = y1 + size
            cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        mask = mask.transpose(2,0,1)
        masks += [mask]
    return torch.Tensor(masks)


def make_noise(imgs, weight = 1., thickness=1, mode='train'):
    weight = 1 - weight
    color = (weight,weight,weight)
    n = 10
    w,h = imgs.shape[2:]
    imgs_noise = []
    imgs = imgs.cpu().numpy()
    for img in imgs:
        img = img.copy()
        mask = np.ones(img.transpose(1,2,0).shape)
        
        if mode=='eval': np.random.seed(3)
        x_list = np.random.randint(0, w-1, (2,n))
        if mode=='eval': np.random.seed(4)
        y_list = np.random.randint(0, h-1, (2,n))
        
        for i in range(n):
            p1 = (x_list[0,i], y_list[0,i])
            p2 = (x_list[1,i], y_list[1,i])
            cv2.line(mask, p1, p2, color, thickness=thickness)
        mask = mask.transpose(2,0,1)
        img = img * mask
        imgs_noise += [img]
    return torch.Tensor(imgs_noise)


def make_semseg(imgs, anno_imgs, class_id=[1,20]):
    imgs, anno_imgs = imgs.cpu().numpy(), anno_imgs.cpu().numpy()
    b,h,w = anno_imgs.shape
    anno_imgs_c3 = anno_imgs.reshape(b,1,h,w).repeat(3, axis=1)
    
    id_min, id_max = class_id
    anno_imgs_semseg = np.where((anno_imgs>=id_min) & (anno_imgs<=id_max), 1, 0)
    imgs_semseg = np.where((anno_imgs_c3>=id_min) & (anno_imgs_c3<=id_max), imgs, 0)
    return torch.Tensor(imgs_semseg), torch.Tensor(anno_imgs_semseg).long()


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def make_style_gram(vgg, style_path, batch_size, device,
                    color_mean=(0.485, 0.456, 0.406), color_std=(0.229, 0.224, 0.225)):
    style_transform = transforms.Compose([
        Normalize_Tensor(color_mean, color_std)
    ])
    style = Image.open(style_path)
    style = style.resize((128, 128), Image.ANTIALIAS)
    style,_ = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1)
    style_features = vgg(style.to(device))
    style_gram = [gram_matrix(y) for y in style_features]
    return style_gram

def get_sytle_img(args, path,
                  color_mean=(0.485, 0.456, 0.406), color_std=(0.229, 0.224, 0.225)):   
    style_transform = transforms.Compose([
        Normalize_Tensor(color_mean, color_std)])
    style = Image.open(path)
    style = style.resize((args.imgsize, args.imgsize), Image.ANTIALIAS)
    style,_ = style_transform(style)
    return style

def show_img(inputs,nrow,title=None, filename='output'):
    """Imshow for Tensor."""
    inputs = torchvision.utils.make_grid(inputs, nrow=nrow, padding=0)
    inputs = inputs.detach().numpy().transpose((1,2,0))
    inputs = inputs.astype(np.float32)
    plt.figure(figsize=(15,15))
    plt.axis("off")
    if title: plt.title(title)
    
    #plt.imsave('/home/yanai-lab/takeda-m/Desktop/figure.png', inputs)
    plt.imsave('./outputs/{}.png'.format(filename), inputs)
    
def save_all(imgs, batch, idx, dirname=None, filename=None):
    if dirname!=None:
        if not os.path.exists('./outputs/{}'.format(dirname)):
            os.mkdir('./outputs/{}'.format(dirname))
        for n,img in enumerate(DeNormalize(imgs)):
            img = img.numpy().transpose((1,2,0))
            img = img.astype(np.float32)
            savepath = './outputs/{}/{}'.format(dirname, filename)
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            plt.imsave('{}/{:0=5}.png'.format(savepath, (idx*batch)+n), img)
    
def im2mask(imgs, mask, alpha=0.5, class_id=[1,20]):
    imgs = imgs.cpu().numpy()
    mask = mask.cpu().numpy()
    batch,c,h,w = imgs.shape
    r,g,b = mask[:,0,:,:], mask[:,1,:,:], mask[:,2,:,:]
    r,g,b = abs(r), abs(g), abs(b)
    mask = np.where((r<=alpha)&(g<=alpha)&(b<=alpha), 0, 1)
    mask = mask.reshape(batch,1,h,w).repeat(3, axis=1)
    imgs_masked = imgs * mask
    return torch.Tensor(imgs_masked)

def im2annmask(mask, alpha=0.5, class_id=[1,20]):
    mask = mask.cpu().numpy()
    r,g,b = mask[:,0,:,:], mask[:,1,:,:], mask[:,2,:,:]
    r,g,b = abs(r), abs(g), abs(b)
    mask = np.where((r<=alpha)&(g<=alpha)&(b<=alpha), 0, 1)
    return mask

def resize_taskvec(taskvec, batch, imgsize, mode_cond):
    if mode_cond=='vec':
        return taskvec.unsqueeze(0).repeat(batch,1)
    elif mode_cond=='map':
        return taskvec.unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(batch,1, 1, 1)
    #return taskvec.unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(batch,1,imgsize,imgsize)  