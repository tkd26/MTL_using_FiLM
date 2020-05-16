# -*- coding: utf-8 -*-
import os, sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# proxy
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import tensorboardX as tbx

from utils.dataloader import make_dataloaders
from utils.dataloader_style import make_style_dataloader
from utils.util import *
from utils.util_eval import *
from utils.loss import CrossEntropy2d
from utils.tasks import *

from utils.model import Model
from utils.model_IndividualLastLayer import Model as Model_IndLastLayer
from utils.model_EncIN import Model as Model_EncIN
from utils.model_SharedEnc import Model as Model_SharedEnc
from utils.model_sym import SymModel 
from utils.model_v2 import Model as Model_v2
from utils.model_Add1Conv import Model as Model_Add1Conv

from utils.vgg import Vgg16
from utils.discriminator import Discriminator

from utils.ssim import ssim as calculate_ssim
from utils.mIoU import mean_IU as calculate_IoU
from utils.fid import calculate_fid


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#COLOR_MEAN, COLOR_STD = (0, 0, 0), (1, 1, 1)
COLOR_MEAN, COLOR_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_args():
    parser = argparse.ArgumentParser(description='multi task learning',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inpainting', action='store_true')
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--semseg', action='store_true')
    parser.add_argument('--styletransfer1', action='store_true')
    parser.add_argument('--styletransfer2', action='store_true')
    parser.add_argument('--mix', action='store_true')
    
    parser.add_argument('--no_recon', action='store_true')
    parser.add_argument('--no_inp_adv', action='store_true')
    parser.add_argument('--no_semseg_adv', action='store_true')
    
    parser.add_argument('--mode', default='train', choices=['train', 'val'])
    parser.add_argument('--mode_model', default='Ours', choices=['Single', 'SharedEnc', 'Ours', 'IndLastLayer', 'Ours_beta', 'EncIN', 'Add1Conv', 'SGN', 'SGN_bias'])
    parser.add_argument('--mode_cond', default='vec', choices=['vec', 'map'])
    parser.add_argument('--mode_mix', default='None', choices=['None', '1', '2', '3', '4', '5', '6'])
    # if Ours or EncIN
    parser.add_argument('--fc', default=None, type=int, choices=[None, 1, 3, 5])
    parser.add_argument('--fc_nc', default=64, type=int, choices=[None, 64, 128])
    
    parser.add_argument('-e', '--epochs', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-c', '--classes', type=int, default=21)
    parser.add_argument('--imgsize', type=int, default=128)
    parser.add_argument('--task_num', type=int, default=5)
    parser.add_argument('--save_iter', type=int, default=1)
    parser.add_argument('--load', default=False, nargs='*')
    parser.add_argument('--version', type=str, default=None)
    
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--save_grad_single', action='store_true')
    parser.add_argument('--save_grad_mix', action='store_true')
    parser.add_argument('--save_grad_stmix', action='store_true')
    parser.add_argument('--save_imgs_idx', type=int, default=None)
    
    return parser.parse_args()

#-------------------------------------------------------------------
# train
#-------------------------------------------------------------------

class Manager():
    def __init__(
        self, args, 
        netG, netD,
        dataloaders_dict,
        style_dataloader,
        criterion, optimizer, weight, 
        device, 
        task_vecs, do_task_list, 
        vgg, style_gram,
        save_model_path):
        
        self.args = args
        self.netG = netG
        self.netD = netD
        self.dataloaders_dict = dataloaders_dict
        self.style_dataloader = style_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.weight = weight
        self.device = device
        self.task_vecs = task_vecs
        self.do_task_list = do_task_list
        self.vgg = vgg
        self.style_gram = style_gram
        self.save_model_path = save_model_path
        
    def epoch(self, epoch):
        train_loss = dict([(task_name, 0) for task_name in self.do_task_list])
        val_loss = dict([(task_name, 0) for task_name in self.do_task_list])
        val_mse = dict([(task_name, 0) for task_name in self.do_task_list])
        val_ssim = dict([(task_name, 0) for task_name in self.do_task_list])
        val_iou = dict([(task_name, 0) for task_name in self.do_task_list])
        val_fid = dict([(task_name, 0) for task_name in self.do_task_list])
        
        train_total_loss = 0.0
        val_total_loss = 0.0
        
        # ------------ train --------------
        if self.args.mode=='train':
            print('-------------')
            print('Epoch {}/{}'.format(epoch, self.args.epochs))
            print('-------------')
            # train
            for task in self.netD.keys():
                self.netD[task].train()
            self.netG.train() 
            print('(train)')
            for train_iter, (datas, style) in enumerate(zip(self.dataloaders_dict['train'], self.style_dataloader)):
                train_batch_loss = self.train(train_iter, datas, style)
                for task in train_batch_loss.keys(): # mix:mix
                    train_total_loss += train_batch_loss[task]
                
            # save networks
            if (epoch) % self.args.save_iter == 0:
                torch.save(
                    self.netG.state_dict(), 
                    '{}/{:0=5}.pth'.format(self.save_model_path, epoch))
                if self.args.inpainting:
                    if not self.args.no_inp_adv:
                        torch.save(
                            self.netD['inpainting'].state_dict(), 
                            '{}/netD_inpainting/{:0=5}.pth'.format(self.save_model_path, epoch)) 
                if self.args.semseg:
                    if not self.args.no_semseg_adv:
                        torch.save(
                            self.netD['semseg'].state_dict(), 
                            '{}/netD_semseg/{:0=5}.pth'.format(self.save_model_path, epoch)) 
        
        # ------------ val --------------
        self.netG.eval() 
        print('(val)')
        if self.args.save_imgs_idx!=None:
            save_imgs_iter = self.args.save_imgs_idx//self.args.batch_size
            save_imgs_iter_idx = self.args.save_imgs_idx - save_imgs_iter
        else:
            save_imgs_iter = None
            save_imgs_iter_idx = None
            
        for val_iter, (datas, style) in enumerate(zip(self.dataloaders_dict['val'], self.style_dataloader)):
            if save_imgs_iter==None:
                val_batch_loss, val_batch_mse, val_batch_ssim, val_batch_iou, val_batch_fid = self.val(
                    val_iter, datas, style, save_imgs_iter_idx)
                for task in val_batch_loss.keys(): # mix:mix
                    val_total_loss += val_batch_loss[task]
                for task in val_batch_mse:
                    val_mse[task] += val_batch_mse[task]
                    val_ssim[task] += val_batch_ssim[task]
                    val_iou[task] += val_batch_iou[task]
                    val_fid[task] += val_batch_fid[task]
                    
            elif val_iter==save_imgs_iter:
                val_loss, val_mse, val_ssim, val_iou, val_fid = self.val(
                    val_iter, datas, style, save_imgs_iter_idx)
                break
          
        if self.args.mode=='train':
            train_total_loss = train_total_loss / len(self.do_task_list) / (train_iter + 1)
        val_total_loss = val_total_loss / len(self.do_task_list) / (val_iter + 1)

        for task in val_mse.keys(): # mix:mix2, mix2, mix3
            val_mse[task] = val_mse[task] / (val_iter + 1)
            val_ssim[task] = val_ssim[task] / (val_iter + 1)
            val_iou[task] = val_mse[task] / (val_iter + 1)
            val_fid[task] = val_mse[task] / (val_iter + 1)
        
        return train_total_loss, val_total_loss, val_mse, val_ssim, val_iou, val_fid
        
        
    def train(self, iteration, datas, style):
        # img and annotation img
        imgs, anno_imgs = datas
        imgs, anno_imgs = imgs.to(self.device), anno_imgs.to(self.device)
        # batch size
        batch = imgs.shape[0]
        # paint
        paint_masks = make_paintmask(imgs).to(self.device)
        imgs_paint = imgs * paint_masks
        anno_imgs_paint = anno_imgs.double() * paint_masks[:,0,:,:].double()
        # noise
        imgs_noise = make_noise(imgs).to(self.device)
        # paint+noise
        imgs_paint_noise = make_noise(imgs_paint).to(self.device)
        # semseg
        imgs_semseg,_ = make_semseg(imgs, anno_imgs, class_id=[1,20])
        imgs_semseg = imgs_semseg.to(self.device)
        # style images for styletransfer3 
        imgs_style3 = style.to(self.device)
        # input pattern
        input_type = [imgs, imgs_paint, imgs_noise]
        # perceptual loss weight
        if 'SGN' in self.args.mode_model:
            content_weight = 1e-3
            style_weight = 1e1
        else:
            content_weight = 1e5
            style_weight = 1e9

        ##### output dict #####

        loss = dict([(task_name, 0) for task_name in self.do_task_list])
        outputs = dict([(task_name, 0) for task_name in self.do_task_list])
        
        ##### trainD #####
        with torch.set_grad_enabled(True):
            if self.args.inpainting:
                if not self.args.no_inp_adv:
                    task_vec = resize_taskvec(self.task_vecs['inpainting'], batch, self.args.imgsize, self.args.mode_cond)
                    errD = do_gan(
                        self.netG, self.netD['inpainting'],
                        input_imgs=imgs_paint, real_imgs=imgs,
                        task_vec=task_vec,
                        criterion=(self.criterion['gan'], self.criterion['identity']),
                        device=self.device,
                        mode ='D'
                    )
                    errD = errD * self.weight['inpainting']

                    self.netG.zero_grad()
                    self.netD['inpainting'].zero_grad()
                    errD.backward()
                    self.optimizer['D_inpainting'].step()

            if self.args.semseg:
                if not self.args.no_semseg_adv:
                    task_vec = resize_taskvec(self.task_vecs['semseg'], batch, self.args.imgsize, self.args.mode_cond)
                    errD = do_gan(
                        self.netG, self.netD['semseg'],
                        input_imgs=imgs, real_imgs=imgs_semseg,
                        task_vec=task_vec,
                        criterion=(self.criterion['gan'], self.criterion['identity']),
                        device=self.device,
                        mode ='D'
                    )
                    errD = errD * self.weight['semseg']

                    self.netG.zero_grad()
                    self.netD['semseg'].zero_grad()
                    errD.backward()
                    self.optimizer['D_semseg'].step()

        ##### trainG #####
        with torch.set_grad_enabled(True):
            if not self.args.no_recon:
                task_vec = resize_taskvec(self.task_vecs['recon'], batch, self.args.imgsize, self.args.mode_cond)
                imgs_recon = random.choice(input_type)
                outputs['recon'], loss['recon'] = do_simple_task(
                    self.netG, 
                    imgs_recon, imgs_recon,
                    task_vec,
                    self.criterion['L2'],
                )
                loss['recon'] *= self.weight['recon']

            if self.args.inpainting:
                task_vec = resize_taskvec(self.task_vecs['inpainting'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['inpainting'], loss1 = do_simple_task(
                    self.netG, 
                    imgs_paint, imgs,
                    task_vec,
                    self.criterion['L2']
                )
                if not self.args.no_inp_adv:
                    _, loss2 = do_gan(
                        self.netG, self.netD['inpainting'],
                        input_imgs=imgs_paint, real_imgs=imgs,
                        task_vec=task_vec,
                        criterion=(self.criterion['gan'], self.criterion['identity']), 
                        device=self.device, 
                        mode='G'
                    )
                    loss['inpainting'] = (loss1*0.998+ loss2*(1-0.998)) * self.weight['inpainting']
                else:
                     loss['inpainting'] = loss1 * self.weight['inpainting']

            if self.args.denoising:
                task_vec = resize_taskvec(self.task_vecs['denoising'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['denoising'], loss['denoising'] = do_simple_task(
                    self.netG, 
                    imgs_noise, imgs,
                    task_vec,
                    self.criterion['L2'],
                )
                loss['denoising'] *= self.weight['denoising']

            if self.args.semseg:
                task_vec = resize_taskvec(self.task_vecs['semseg'], batch, self.args.imgsize, self.args.mode_cond)
                _, loss1 = do_simple_task(
                    self.netG, 
                    imgs, imgs_semseg,
                    task_vec,
                    self.criterion['L2']
                )

                if not self.args.no_semseg_adv:
                    outputs['semseg'], loss2 = do_gan(
                        self.netG, self.netD['semseg'],
                        input_imgs=imgs, real_imgs=imgs_semseg,
                        task_vec=task_vec,
                        criterion=(self.criterion['gan'], self.criterion['identity']), 
                        device=self.device, 
                        mode='G'
                    )
                    loss['semseg'] = (loss1*0.999+ loss2*(1-0.999)) * self.weight['semseg']
                else:
                    loss['semseg'] = loss1 * self.weight['semseg']

            if self.args.styletransfer1:
                task_vec = resize_taskvec(self.task_vecs['styletransfer1'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['styletransfer1'], loss['styletransfer1'] = do_styletrnsfer(
                    self.netG, self.vgg,
                    input_imgs=imgs, style_gram=self.style_gram['styletransfer1'],
                    task_vec=task_vec,
                    criterion=self.criterion['L2'],
                    content_weight=content_weight,
                    style_weight=style_weight,
                )
                loss['styletransfer1'] *= self.weight['styletransfer1']

            if self.args.styletransfer2:
                task_vec = resize_taskvec(self.task_vecs['styletransfer2'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['styletransfer2'], loss['styletransfer2'] = do_styletrnsfer(
                    self.netG, self.vgg,
                    input_imgs=imgs, style_gram=self.style_gram['styletransfer2'],
                    task_vec=task_vec,
                    criterion=self.criterion['L2'],
                    content_weight=content_weight,
                    style_weight=style_weight,
                )
                loss['styletransfer2'] *= self.weight['styletransfer2']

            if self.args.mix:
                if self.args.mode_mix=='1':
                    mix_tasks = {
                        'mix1':{
                            'task':['denoising', 'styletransfer1'],
                            'input':imgs_noise,
                            'target':outputs['styletransfer1'].detach()},
                        'mix2':{
                            'task':['denoising', 'semseg'],
                            'input':imgs_noise,
                            'target':imgs_semseg},
                        'mix3':{
                            'task':['semseg', 'styletransfer1'],
                            'input':imgs,
                            'target':make_semseg(outputs['styletransfer1'].detach(), anno_imgs)[0].to(self.device)},
                    }

                for mix_patern in mix_tasks.keys():
                    mix = mix_tasks[mix_patern]
                    task_vec = (self.task_vecs[mix['task'][0]]+self.task_vecs[mix['task'][1]]).unsqueeze(0).repeat(batch,1)
                    outputs[mix_patern], loss[mix_patern] = do_simple_task(
                        self.netG, 
                        mix['input'], mix['target'],
                        task_vec,
                        self.criterion['L2']
                    )
                
            all_loss = 0.0
            for task_name in self.do_task_list:
                all_loss += loss[task_name]

            self.netG.zero_grad()
            for task in self.netD.keys():
                self.netD[task].zero_grad()
            all_loss.backward()
            self.optimizer['G'].step()
        
        return loss
    
    def val(self, iteration, datas, style, save_imgs_iter_idx):
        # img and annotation img
        imgs, anno_imgs = datas
        imgs, anno_imgs = imgs.to(self.device), anno_imgs.to(self.device)
        # batch size
        batch = imgs.shape[0]
        # paint
        paint_masks = make_paintmask(imgs, mode='val').to(self.device)
        imgs_paint = imgs * paint_masks
        anno_imgs_paint = anno_imgs.double() * paint_masks[:,0,:,:].double()
        # noise
        imgs_noise = make_noise(imgs).to(self.device)
        # paint+noise
        imgs_paint_noise = make_noise(imgs_paint).to(self.device)
        # semseg
        imgs_semseg,_ = make_semseg(imgs, anno_imgs, class_id=[1,20])
        imgs_semseg = imgs_semseg.to(self.device)
        # style images for styletransfer3 
        imgs_style3 = style.to(self.device)
        # input pattern
        input_type = [imgs, imgs_paint, imgs_noise]
        # perceptual loss weight
        if 'SGN' in self.args.mode_model:
            content_weight = 1e-3
            style_weight = 1e1
        else:
            content_weight = 1e5
            style_weight = 1e9

        ##### output dict #####

        mse = dict([(task_name, 0) for task_name in self.do_task_list])
        ssim = dict([(task_name, 0) for task_name in self.do_task_list])
        iou = dict([(task_name, 0) for task_name in self.do_task_list])
        fid = dict([(task_name, 0) for task_name in self.do_task_list])
        loss = dict([(task_name, 0) for task_name in self.do_task_list])
        outputs = dict([(task_name, 0) for task_name in self.do_task_list])

        ##### trainG #####
        with torch.set_grad_enabled(False):
            if not self.args.no_recon:
                task_vec = resize_taskvec(self.task_vecs['recon'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['recon'], loss['recon'] = do_simple_task(
                    self.netG, 
                    imgs, imgs,
                    task_vec,
                    self.criterion['L2'],
                )
                loss['recon'] *= self.weight['recon']
                mse['recon'] = loss['recon']
                #ssim['recon'] = calculate_ssim(outputs['recon'],imgs)

            if self.args.inpainting:
                task_vec = resize_taskvec(self.task_vecs['inpainting'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['inpainting'], loss['inpainting'] = do_simple_task(
                    self.netG, 
                    imgs_paint, imgs,
                    task_vec,
                    self.criterion['L2']
                )
                loss['inpainting'] *= self.weight['inpainting']
                mse['inpainting'] = loss['inpainting']
                #ssim['inpainting'] = calculate_ssim(outputs['inpainting'],imgs)

            if self.args.denoising:
                task_vec = resize_taskvec(self.task_vecs['denoising'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['denoising'], loss['denoising'] = do_simple_task(
                    self.netG, 
                    imgs_noise, imgs,
                    task_vec,
                    self.criterion['L2'],
                )
                loss['denoising'] *= self.weight['denoising']
                mse['denoising'] = loss['denoising']
                #ssim['denoising'] = calculate_ssim(outputs['denoising'],imgs)

            if self.args.semseg:
                task_vec = resize_taskvec(self.task_vecs['semseg'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['semseg'], loss['semseg'] = do_simple_task(
                    self.netG, 
                    imgs, imgs_semseg,
                    task_vec,
                    self.criterion['L2']
                )
                loss['semseg'] *= self.weight['semseg']
                mse['semseg'] = loss['semseg']
                #ssim['semseg'] = calculate_ssim(outputs['semseg'],imgs_semseg)
                #iou['semseg'] = calculate_IoU(im2annmask(outputs['semseg']),im2annmask(imgs_semseg))

            if self.args.styletransfer1:
                task_vec = resize_taskvec(self.task_vecs['styletransfer1'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['styletransfer1'] = self.netG(imgs, task_vec)
                #fid['styletransfer1'] = calculate_fid(imgs, outputs['styletransfer1'].detach(), False, batch)

            if self.args.styletransfer2:
                task_vec = resize_taskvec(self.task_vecs['styletransfer2'], batch, self.args.imgsize, self.args.mode_cond)
                outputs['styletransfer2'] = self.netG(imgs, task_vec)
                #fid['styletransfer2'] = calculate_fid(imgs, outputs['styletransfer2'].detach(), False, batch)

            if self.args.mix:
                if self.args.mode_mix=='1':
                    mix_tasks = {
                        'mix1':{
                            'task':['denoising', 'styletransfer1'],
                            'input':imgs_noise,
                            'target':outputs['styletransfer1'].detach()},
                        'mix2':{
                            'task':['denoising', 'semseg'],
                            'input':imgs_noise,
                            'target':imgs_semseg},
                        'mix3':{
                            'task':['semseg', 'styletransfer1'],
                            'input':imgs,
                            'target':make_semseg(outputs['styletransfer1'].detach(), anno_imgs)[0].to(self.device)},
                    }
                mix_outputs = []
                #mix_losses = []
                all_task = []
                for mix_patern in mix_tasks.keys():
                    mix = mix_tasks[mix_patern]
                    task_vec = (self.task_vecs[mix['task'][0]]+self.task_vecs[mix['task'][1]]).unsqueeze(0).repeat(batch,1)
                    outputs[mix_patern], loss[mix_patern] = do_simple_task(
                        self.netG, 
                        mix['input'], mix['target'],
                        task_vec,
                        self.criterion['L2']
                    )
                    
                    mse[mix_patern] = loss[mix_patern]
                    if 'styletransfer' in mix['task'][0] or 'styletransfer' in mix['task'][1]:
                        pass
                        #fid[mix_patern] = calculate_fid(imgs, mix_output.detach(), False, batch)
                    else:
                        pass
                        #ssim[task_name] = calculate_ssim(mix_output, combi['target'])
                    if 'semseg' in mix['task'][0] or 'semseg' in mix['task'][1]:
                        pass
                        #iou[mix_patern] = calculate_IoU(im2annmask(mix_output.detach()),im2annmask(imgs_semseg))
                        
                    if self.args.save_grad_mix:
                        gradually_mix(
                            self.task_vecs, batch, self.netG, mix['task'], 
                            mix['input'].to(self.device), mix['target'].to(self.device), self.criterion['L1'],
                            idx=save_imgs_iter_idx
                        )
                    mix_outputs.append(outputs[mix_patern])
                        
                if self.args.save_outputs:
                    outputs['mix'] = torch.cat([mix_output for mix_output in mix_outputs], dim=0)
                    filename = list(set(all_task))
                    filename = 'mix_' + '_'.join(filename)
                    show_img(DeNormalize(outputs['mix']), nrow=batch, filename=filename)
                    
            if self.args.save_outputs:  
                imgs_original = DeNormalize(imgs)
                view_targets = [imgs_paint, imgs_noise, imgs_semseg]
                single_task_list = [s for s in self.do_task_list if '+' not in s]
                output_all_task = torch.cat([DeNormalize(outputs[task_name]) for task_name in single_task_list], dim=0)
                if len(view_targets) > 0:
                    view_targets = torch.cat([DeNormalize(i) for i in view_targets], dim=0).cpu()
                    show_imgs = torch.cat([imgs_original, view_targets, output_all_task], dim=0)
                else:
                    show_imgs = torch.cat([imgs_original, output_all_task], dim=0)

                show_img(show_imgs, nrow=batch, filename='outputs')
                
            if self.args.save_grad_stmix:
                gradually_mix_styletransfer(
                    self.task_vecs, batch, self.netG, ['styletransfer1', 'styletransfer2'], 
                    imgs.to(self.device), imgs.to(self.device), self.criterion['L2'], idx=save_imgs_iter_idx)
                
            if self.args.save_grad_single:
                imgs_input = {
                    'inpainting':imgs_paint,
                    'denoising':imgs_noise,
                    'semseg':imgs,
                    'styletransfer1':imgs, 
                    'styletransfer2':imgs,
                }
                gradually_single(task_vecs, batch, netG, do_task_list, imgs_input, imgs, criterion, idx=save_imgs_iter_idx)
                
                #save_all(imgs, args.batch_size, iteration, dirname='normal', filename='input_original')
                
        return loss, mse, ssim, iou, fid



#-------------------------------------------------------------------
# main
#-------------------------------------------------------------------

def main(args):
    
    ##### Data Loader #####
  
    rootpath = '/export/space0/takeda-m/benchmark_RELEASE/dataset/'
    #rootpath = "/export/data/pascal/VOC2012/VOCdevkit/VOC2012/" # normal dataset path

    train_dataloader, val_dataloader = make_dataloaders(
        args, rootpath, color_mean=COLOR_MEAN, color_std=COLOR_STD)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        
    # number of images
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    print('train num:{}, val num:{}'.format(num_train_imgs, num_val_imgs))
    
    style_dataloader = make_style_dataloader(args, num_train_imgs)
    
    ##### train #####
    
    task_dict = {
        'recon': not args.no_recon,
        'inpainting':args.inpainting, 
        'denoising':args.denoising, 
        'semseg':args.semseg, 
        'styletransfer1':args.styletransfer1, 
        'styletransfer2':args.styletransfer2,
        'mix1':args.mix,
        'mix2':args.mix,
        'mix3':args.mix,
    }
    do_task_list = [t for t,val in task_dict.items() if val==True]
    print(args.mode_model)
    print(do_task_list)
    save_name = []
    save_name.extend(['COND'+args.mode_cond])
    save_name.extend(['MODE'+args.mode_model])
    
    if args.fc!=None: save_name.extend(['FC{}'.format(args.fc)])
    if args.fc_nc!=None: save_name.extend([str(args.fc_nc)])
    if args.mode_mix!='None': save_name.extend(['MIX{}'.format(args.mode_mix)])
        
    if args.no_inp_adv: save_name.extend(['NoInpAdv'])
    if args.no_semseg_adv: save_name.extend(['NoSemsegAdv'])
        
    if args.version: save_name.extend([args.version])
        
    save_name.extend(do_task_list)
    save_name = '_'.join(save_name)
    save_model_path = './weights/{}'.format(save_name)
    save_log_path = './logs/{}'.format(save_name)
    
    # define writer
    if args.mode=='train':
        writer = tbx.SummaryWriter(log_dir=save_log_path)
    
    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    # model
    if 'SGN' in args.mode_model:
        netG = SymModel(3, 3, args.task_num, args.mode_model=='SGN_bias')
    elif args.mode_model == 'EncIN':
        netG = Model_EncIN(3, 3, args.task_num, args.mode_cond)
    elif args.mode_model == 'Single':
        netG = Model_v2(3, 3, args.task_num, 'single', args.fc, args.fc_nc)
    elif args.mode_model == 'SharedEnc':
        netG = Model_SharedEnc(3, 3, args.task_num, args.mode_cond)
    elif args.mode_model == 'Ours':
        netG = Model_v2(3, 3, args.task_num, args.mode_cond, args.fc, args.fc_nc)
    elif args.mode_model == 'IndLastLayer':
        netG = Model_IndLastLayer(3, 3, args.task_num, args.mode_cond, args.fc, args.fc_nc)
    elif args.mode_model == 'Ours_beta':
        netG = Model(3, 3, args.task_num, args.mode_cond)
    elif args.mode_model == 'Add1Conv':
        netG = Model_Add1Conv(3, 3, args.task_num, args.mode_cond)
    else:
        sys.exit('Error:No model selected')
            
    netG.to(device)
    netG = torch.nn.DataParallel(netG)
    
    # count parameters
    param = 0
    for p in netG.parameters():
        param += p.numel()
    print('param:{}'.format(param))
    
    netD = {
        'inpainting':Discriminator().to(device),
        'semseg':Discriminator().to(device),
    }
    for task in netD.keys():
        netD[task] = torch.nn.DataParallel(netD[task])
        
    # load parameter
    if args.load:
        load_path, load_epoch = args.load
        load_epoch = int(load_epoch)
        param = torch.load('{}/{:0=5}.pth'.format(load_path, load_epoch))
        netG.load_state_dict(param)
        
        if args.mode == 'train':
            if args.inpainting:
                if not args.no_inp_adv:
                    param = torch.load('{}/netD_inpainting/{:0=5}.pth'.format(load_path, load_epoch))
                    netD['inpainting'].load_state_dict(param)
            if args.semseg:
                if not args.no_semseg_adv:
                    param = torch.load('{}/netD_semseg/{:0=5}.pth'.format(load_path, load_epoch))
                    netD['semseg'].load_state_dict(param)
    else:
        load_epoch = 0
        
    # make folder to save network
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if args.inpainting:
        if not args.no_inp_adv:
            if not os.path.exists(os.path.join(save_model_path, 'netD_inpainting')):
                os.mkdir(os.path.join(save_model_path, 'netD_inpainting'))
    if args.semseg:
        if not args.no_semseg_adv:
            if not os.path.exists(os.path.join(save_model_path, 'netD_semseg')):
                os.mkdir(os.path.join(save_model_path, 'netD_semseg'))
    
    torch.backends.cudnn.benchmark = True
    
    # make style gram matrix
    vgg = Vgg16(requires_grad=False).to(device)
    style_gram1 = make_style_gram(
        vgg, style_path='./style_images/starry_night.jpg',batch_size=args.batch_size, device=device)
    style_gram2 = make_style_gram(
        vgg, style_path='./style_images/the_scream.jpg',batch_size=args.batch_size, device=device)
    
    style_gram = {'styletransfer1':style_gram1, 'styletransfer2':style_gram2}
    
    # train parameter
    optimizer = {
        'G':torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.1, 0.999)),
        'D_inpainting':torch.optim.Adam(netD['inpainting'].parameters(), lr=1e-4, betas=(0.1, 0.999)),
        'D_semseg':torch.optim.Adam(netD['semseg'].parameters(), lr=1e-4, betas=(0.1, 0.999)),
    }
    
    criterion = { 
        'L2':nn.MSELoss(),
        'gan':nn.BCELoss(),
        'identity':nn.L1Loss(),
    }
    
    '''
    weight = { 
        'recon':1.0,
        'inpainting':7.0,
        'denoising':7.0,
        'semseg':5.0,
        'styletransfer1':1.0e-7,
        'styletransfer2':1.0e-7,
        'mix1':2.0,
        'mix2':2.0,
        'mix3':2.0,
    }
    '''
    weight = { 
        'recon':1.0,
        'inpainting':1.0,
        'denoising':1.0,
        'semseg':1.0,
        'styletransfer1':1.0e-7,
        'styletransfer2':1.0e-7,
        #'styletransfer3':1.0e-1,
        'mix1':1.0,
        'mix2':1.0,
        'mix3':1.0,
    }
    if 'SGN' in args.mode_model:
        for task in weight.keys():
            weight[task] = 1.0
    
    # task vectors
    if args.task_num==5:
        task_vecs = { 
            'recon':torch.Tensor([0,0,0,0,0]),
            'inpainting':torch.Tensor([1,0,0,0,0]),
            'denoising':torch.Tensor([0,1,0,0,0]),
            'semseg':torch.Tensor([0,0,1,0,0]),
            'styletransfer1':torch.Tensor([0,0,0,1,0]),
            'styletransfer2':torch.Tensor([0,0,0,0,1]),
        }
    elif args.task_num==6:
        task_vecs = { 
            'recon':torch.Tensor([0,0,0,0,0,0]),
            'inpainting':torch.Tensor([1,0,0,0,0,0]),
            'denoising':torch.Tensor([0,1,0,0,0,0]),
            'semseg':torch.Tensor([0,0,1,0,0,0]),
            'styletransfer1':torch.Tensor([0,0,0,1,0,0]),
            'styletransfer2':torch.Tensor([0,0,0,0,1,0]),
            'styletransfer3':torch.Tensor([0,0,0,0,0,1]),
        }
    
    manager = Manager(
        args,
        netG, netD,
        dataloaders_dict, style_dataloader,
        criterion, optimizer, weight,
        device,
        task_vecs, do_task_list,
        vgg, style_gram,
        save_model_path)
    
    for epoch in range(load_epoch+1, args.epochs+1):
        train_total_loss, val_total_loss, val_mse, val_ssim, val_iou, val_fid = manager.epoch(epoch)
        if args.mode=='val': break
        else:
            # write tensorboardX
            writer.add_scalars('total_loss',
                               {'train': train_total_loss,
                                'val': val_total_loss},
                                   epoch)
            writer.add_scalars('val_mse', val_mse, epoch)
            #writer.add_scalars('val_ssim', val_ssim, epoch)
            #writer.add_scalars('val_iou', val_iou, epoch)
            #writer.add_scalars('val_fid', val_fid, epoch)
                    
    if args.mode=='train': writer.close()

if __name__ == '__main__':
    args = get_args()
    main(args)