import numpy as np
import torch

def iou(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):
        #for cls in range(n_classes-1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


def acc(pred, target, n_classes):
    acc = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):
        #for cls in range(n_classes-1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
        if target_inds.long().sum().data.cpu()[0] == 0:
            acc.append(float(0))
        else:
            acc.append(float(intersection) / float(target_inds.long().sum().data.cpu()[0]))
    return np.sum(np.array(acc)) / n_classes



def gradually_mix(task_vecs, batch, netG,
                  task_pair, imgs_input, imgs_target, criterion,
                  idx=0):
    taskA,taskB = task_pair
    outputs = []
    for i in range(6):
        for j in range(6):
            task_vec = (task_vecs[taskA]*(i*0.2)+task_vecs[taskB]*(j*0.2)).unsqueeze(0).repeat(1,1)
            output, loss = do_simple_task(
                netG, 
                imgs_input[idx].unsqueeze(0), imgs_target[idx].unsqueeze(0),
                task_vec,
                criterion
            )
            outputs += [output.detach()]
    outputs = torch.cat([DeNormalize(i) for i in outputs], dim=0).cpu()
    show_img(outputs, nrow=6, filename='{}_{}'.format(taskA,taskB), show=False, save=True)
    
def gradually_mix_styletransfer(task_vecs, batch, netG,
                  task_pair, imgs_input, imgs_target, criterion,
                  idx=0):
    taskA,taskB = task_pair
    outputs = []
    for i in [5]:
        task_vec = (task_vecs[taskA]*(1-(i*0.1))+task_vecs[taskB]*(i*0.1) ).unsqueeze(0).repeat(1,1)
        output, loss = do_simple_task(
            netG, 
            imgs_input[idx].unsqueeze(0), imgs_target[idx].unsqueeze(0),
            task_vec,
            criterion
        )
        outputs += [output.detach()]
    outputs = torch.cat([DeNormalize(i) for i in outputs], dim=0).cpu()
    show_img(outputs, nrow=1, filename='{}_{}'.format(taskA,taskB), show=False, save=True)
    
def gradually_single(task_vecs, batch, netG,
                  do_task_list, imgs_input, imgs_target, criterion,
                  idx=0):
    outputs = []
    for task in do_task_list:
        if 'mix' in task:
            do_task_list.remove(task)
    nrow = len(do_task_list)
    for i in [0,2,4,6,8,10]:
        for task in do_task_list:
            if task=='mix': continue
            if task=='recon': continue
            task_vec = (task_vecs[task]*(i*0.1)).unsqueeze(0).repeat(1,1)
            output, loss = do_simple_task(
                netG, 
                imgs_input[task][idx].unsqueeze(0), imgs_target[idx].unsqueeze(0),
                task_vec,
                criterion[task]
            )
            outputs += [output.detach()]
    outputs = torch.cat([DeNormalize(i) for i in outputs], dim=0).cpu()
    show_img(outputs, nrow=5, filename='gradually_single', show=False, save=True)
    
    

'''
def gradually_mix3(batch, netG,
                  tasks, imgs_input, imgs_target, criterion,
                  idx=0):
    task_vec_list = [
        torch.Tensor([0,0,0,0,0,0]),
        torch.Tensor([0,0.5,0.5,0.5,0,0]),
        torch.Tensor([0,1,0.5,0.5,0,0]),
        torch.Tensor([0,0.5,1,0.5,0,0]),
        torch.Tensor([0,0.5,0.5,1,0,0]),
        torch.Tensor([0,1,1,0.5,0,0]),
        torch.Tensor([0,1,0.5,1,0,0]),
        torch.Tensor([0,0.5,1,1,0,0]),
        torch.Tensor([0,1,1,1,0,0]),
    ]
    taskA,taskB,taskC = tasks
    outputs = []
    for task_vec in task_vec_list:
            task_vec = task_vec.unsqueeze(0).repeat(1,1)
            output, loss = do_simple_task(
                netG, 
                imgs_input[idx].unsqueeze(0), imgs_target[idx].unsqueeze(0),
                task_vec,
                criterion
            )
            outputs += [output.detach()]
    outputs = torch.cat([DeNormalize(i) for i in outputs], dim=0).cpu()
    show_img(outputs, nrow=len(task_vec_list), filename='{}_{}_{}'.format(taskA,taskB,taskC), show=False, save=True)
'''





