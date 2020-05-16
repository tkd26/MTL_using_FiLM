import numpy as np
import torch
from .util import gram_matrix

def do_simple_task(net, 
                   input_imgs, targets, 
                   task_vec, 
                   criterion):
    #task_vec = task_vec.unsqueeze(0).repeat(input_imgs.shape[0],1)
    outputs = net(input_imgs, task_vec)
    loss = criterion(outputs, targets)
    return outputs, loss


def do_gan(netG, netD,
           input_imgs, real_imgs,
           task_vec,
           criterion,
           device,
           mode):
    #task_vec = task_vec.unsqueeze(0).repeat(input_imgs.shape[0],1)
    criterion_gan, criterion_id = criterion
    outputs = netG(input_imgs, task_vec).to(device)
            
    if mode=='D':
        # train with real
        D_real = netD(real_imgs).to(device)
        errD_real = criterion_gan(D_real, torch.tensor(1.0).expand_as(D_real).to(device))
        # train with fake
        D_fake = netD(outputs).to(device)
        errD_fake = criterion_gan(D_fake, torch.tensor(0.0).expand_as(D_fake).to(device))
        errD = (errD_real + errD_fake) / 2
        return errD
        
    if mode=='G':
        D_fake = netD(outputs).to(device)
        errG_gan = criterion_gan(D_fake, torch.tensor(1.0).expand_as(D_fake).to(device)) 
        errG_id = criterion_id(netG(real_imgs, task_vec), real_imgs)
        loss = errG_gan + errG_id * 5
        return outputs, loss
    
def do_cyclegan(netG, netD_A, netD_B,
           real_A, real_B,
           task_vec_A2B, task_vec_B2A,
           criterion,
           device,
           mode):
    
    criterion_gan, criterion_id, criterion_cycle = criterion
    fake_A = netG(real_B, task_vec_B2A)
    fake_B = netG(real_A, task_vec_A2B)
    target_real = torch.Tensor(real_A.shape[0]).fill_(1.0).to(device)
    target_fake = torch.Tensor(real_A.shape[0]).fill_(0.0).to(device)
            
    if mode=='D_A':
        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_gan(pred_real, target_real)
        # Fake loss
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_gan(pred_fake, target_fake)
        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        return loss_D_A
    
    if mode=='D_B':
        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_gan(pred_real, target_real)
        # Fake loss
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_gan(pred_fake, target_fake)
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        return loss_D_B
        
    if mode=='G':
        # identity loss
        same_B = netG(real_B, task_vec_A2B).to(device)
        loss_id_B = criterion_id(same_B, real_B)*5.0
        same_A = netG(real_A, task_vec_B2A).to(device)
        loss_id_A = criterion_id(same_A, real_A)*5.0
        
        # GAN loss
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_gan(pred_fake, target_real)
        
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_gan(pred_fake, target_real)
        
        # Cycle loss
        recovered_A = netG(fake_B, task_vec_B2A)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        
        recovered_B = netG(fake_A, task_vec_A2B)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
        # Total loss
        loss_G = loss_id_A + loss_id_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        
        return (fake_A, fake_B), loss_G
    
    
def do_styletrnsfer(net, vgg, 
                    input_imgs, style_gram,
                    task_vec,
                    criterion, 
                    content_weight=1e5, style_weight=5e9):
    #task_vec = task_vec.unsqueeze(0).repeat(input_imgs.shape[0],1)
    y = net(input_imgs, task_vec)
    y_features = vgg(y)
    x_features = vgg(input_imgs)
    # content_loss
    Lcontent = content_weight * criterion(y_features.relu2_2, x_features.relu2_2)
    # style loss
    style_weights = np.array([1e-1, 1, 1e1, 5]) * style_weight
    Lstyle = 0.
    for y_ft, s_gm, weight in zip(y_features, style_gram, style_weights):
        y_gm = gram_matrix(y_ft)
        Lstyle += criterion(y_gm, s_gm[:input_imgs.shape[0], :, :]) * weight 
    
    loss = Lcontent + Lstyle
    return y, loss