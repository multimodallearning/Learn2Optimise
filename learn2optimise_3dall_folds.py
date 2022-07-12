#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import nibabel as nib
import os
import sys


#!nvidia-smi

from l2r_heatmorph_utils import *


# In[53]:


from scipy.ndimage import distance_transform_edt as edt
H = 192
W = 192
D = 208

import csv
import struct
import time
from torch.utils.checkpoint import checkpoint 


def get_data():
    keypts_all_mov = []
    keypts_all_fix = []

    mind_ch = 12

    mind_all_mov = torch.zeros(28,mind_ch,H//2,W//2,D//2).pin_memory()
    mind_all_fix = torch.zeros(28,mind_ch,H//2,W//2,D//2).pin_memory()

    img_all_mov = torch.zeros(28,1,H,W,D).pin_memory()
    img_all_fix = torch.zeros(28,1,H,W,D).pin_memory()
    mask_all_mov = torch.zeros(28,1,H,W,D).pin_memory()
    mask_all_fix = torch.zeros(28,1,H,W,D).pin_memory()

    time_mind = 0

    for ii,i in enumerate((1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,7,8,14,18,20,21,28)):
        if((ii>=10)&(ii<20)): #4DCT
        
            folder = 'learn2optimise_4dct/'; dat = '_F_insp.dat';
            zfill = 3
            mask_all_fix[ii] = torch.from_numpy(nib.load('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_mask_exh.nii.gz').get_fdata()).float()
            mask_all_mov[ii] = torch.from_numpy(nib.load('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_mask_inh.nii.gz').get_fdata()).float()

            img_all_fix[ii] = torch.from_numpy(nib.load('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_img_exh.nii.gz').get_fdata()).float()
            img_all_mov[ii] = torch.from_numpy(nib.load('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_img_inh.nii.gz').get_fdata()).float()
            #img_all_fix[ii] += 1000
            #img_all_mov[ii] += 1000

            corrfield_exh = torch.empty(0,3)
            with open('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_kpts_exh.csv', newline='') as csvfile:
                fread = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in fread:
                    corrfield_exh = torch.cat((corrfield_exh,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)
            corrfield_insp = torch.empty(0,3)
            with open('learn2optimise_4dct/4DCT_'+str(i).zfill(2)+'_kpts_inh.csv', newline='') as csvfile:
                fread = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in fread:
                    corrfield_insp = torch.cat((corrfield_insp,torch.from_numpy(np.array(row).astype('float32')).float().view(1,3)),0)

            corrfield = torch.cat((corrfield_insp,corrfield_exh),1)
            keypts_mov = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1).cuda()
            keypts_fix = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1).cuda()


        if(ii<10): #COPD
            folder = 'COPDgene/'; dat = '_insp.dat';
            mask_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_exp_mask.nii.gz').get_fdata()).float()
            mask_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_insp_mask.nii.gz').get_fdata()).float()
            zfill = 2
            img_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_exp.nii.gz').get_fdata()).float()
            img_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_insp.nii.gz').get_fdata()).float()
            img_all_fix[ii] += 1000
            img_all_mov[ii] += 1000
            with open(folder+'/keypoints/case_'+str(i).zfill(2)+dat, 'rb') as content_file:
                content = content_file.read()
            corrfield = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(-1,6).float()
            keypts_mov = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1).cuda()
            keypts_fix = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1).cuda()


        if(ii>=20): #EMPIRE
            folder = 'EMPIRE10/'; dat = '.dat';
            mask_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_exp_mask.nii.gz').get_fdata()).float()
            mask_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_insp_mask.nii.gz').get_fdata()).float()
            zfill = 2
            img_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_exp.nii.gz').get_fdata()).float()
            img_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_insp.nii.gz').get_fdata()).float()
            img_all_fix[ii] += 1000
            img_all_mov[ii] += 1000

            with open(folder+'/keypoints/case_'+str(i).zfill(2)+dat, 'rb') as content_file:
                content = content_file.read()
            corrfield = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(-1,6).float()
            keypts_fix = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1).cuda()
            keypts_mov = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1).cuda()


        keypts_all_mov.append(keypts_mov)
        keypts_all_fix.append(keypts_fix)
        grid_sp = 2

        torch.cuda.synchronize()
        t0 = time.time()
        fixed_mask = mask_all_fix[ii].view(1,1,H,W,D).cuda()
        moving_mask = mask_all_mov[ii].view(1,1,H,W,D).cuda()

        fixed = img_all_fix[ii].view(1,1,H,W,D).cuda()
        moving = img_all_mov[ii].view(1,1,H,W,D).cuda()

        #compute MIND descriptors and downsample (using average pooling)
        with torch.no_grad():
            with torch.cuda.amp.autocast():

                ##replicate masking!!!
                #avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
                #avg3.cuda()
                #mask = (avg3(fixed_mask.view(1,1,H,W,D).cuda())>0.9).float()
                #dist,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
                #fixed_r = F.interpolate((fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*104*96+idx[1]*104+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
                #fixed_r.view(-1)[mask.view(-1)!=0] = fixed.cuda().reshape(-1)[mask.view(-1)!=0]
                #mask = (avg3(moving_mask.view(1,1,H,W,D).cuda())>0.9).float()
                #dist,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
                #moving_r = F.interpolate((moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*104*96+idx[1]*104+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
                #moving_r.view(-1)[mask.view(-1)!=0] = moving.cuda().reshape(-1)[mask.view(-1)!=0]


                #compute MIND descriptors and downsample (using average pooling)
                #with torch.no_grad():
                #mindssc_fix = MINDSSC(fixed_r,1,2).half()#*fixed_mask.cuda().half()#.cpu()
                #mindssc_mov = MINDSSC(moving_r,1,2).half()#*moving_mask.cuda().half()#.cpu()

                #mind_fix = F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)
                #mind_mov = F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp)



                mind_fix_ = mask_all_fix[ii].view(1,1,H,W,D).cuda().half()*            MINDSSC(img_all_fix[ii].view(1,1,H,W,D).cuda(),1,2).half()
                mind_mov_ = mask_all_mov[ii].view(1,1,H,W,D).cuda().half()*            MINDSSC(img_all_mov[ii].view(1,1,H,W,D).cuda(),1,2).half()
                mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)
                mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)
        torch.cuda.synchronize()
        t1 = time.time()
        time_mind += (t1-t0)
        mind_all_fix[ii] = mind_fix.cpu()
        mind_all_mov[ii] = mind_mov.cpu()
    print('mind computation',time_mind/30,'sec') 


    # In[61]:


    #pred_data = torch.load('keypoints_lung_ct_all.pth')
    #keypts_all_pred = pred_data['keypts_all_pred'][8:18]
    #for i in range(10):
    #    keypts_all_pred.append(torch.zeros_like(keypts_all_fix[i+10]))
    #for i in range(8):
    #    keypts_all_pred.append(pred_data['keypts_all_pred'][i])

    pred_data = torch.load('keypoints_lung_ct_all.pth')
    pred_4dct = torch.load('keypt_pred_lung_ct_4dct.pth')
    keypts_all_pred = pred_data['keypts_all_pred'][8:18]
    for i in range(10):
        keypts_all_pred.append(pred_4dct[i])#torch.zeros_like(keypts_all_fix[i+10]))
    for i in range(8):
        keypts_all_pred.append(pred_data['keypts_all_pred'][i])
    
    idx = torch.arange(17,18)
    keypts_fix = keypts_all_fix[int(idx)].cuda()[:]
    keypts_mov = keypts_all_mov[int(idx)].cuda()[:]

    pred_xyz = keypts_all_pred[int(idx)].cuda()#(keypts_mov-keypts_fix)
    return keypts_all_fix,keypts_all_mov,keypts_all_pred,mind_all_fix,mind_all_mov




def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))
#gpu_usage()


if __name__ == "__main__":
    gpu_id = 0
    fold_nu = 1
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"Fold {i:>6}: {arg}")
            fold_nu  = int(arg)
        else:
            if(i==2):
                print(f"GPU ID {i:>6}: {arg}")
                gpu_id = int(arg)
            else:
                print(f"Argument {i:>6}: {arg}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(torch.cuda.get_device_name())

    keypts_all_fix,keypts_all_mov,keypts_all_pred,mind_all_fix,mind_all_mov = get_data()
    ident = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H//2,W//2,D//2),align_corners=True).permute(0,4,1,2,3).cuda()
    H=192;W=192;D=208
    fixed_off = 0.015*F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,2,2,2),align_corners=True).permute(0,4,1,2,3).view(1,3,8,1,1).repeat(1,1,1,(H//2)*(W//2)*(D//2),1)



    class ConvBlock(nn.Module):
        def __init__(self, ndims, in_channels, out_channels, stride=1):
            super().__init__()
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.main = Conv(in_channels//2, out_channels//2, 3, stride, 1)
            self.norm = nn.InstanceNorm3d(out_channels//2)
            self.activation = nn.ReLU()#nn.LeakyReLU(0.2)
            self.main2 = Conv(out_channels//2, out_channels//2, 1, stride,0)
            self.norm2 = nn.InstanceNorm3d(out_channels//2)
            self.activation2 = nn.ReLU()#nn.LeakyReLU(0.2)
        def forward(self, x):
            out = self.activation(self.norm(self.main(x)))
            out = self.activation2(self.norm2(self.main2(out)))
            return out
    
    inshape = (224//2,224//2,224//2)
    nb_unet_features=None; nb_unet_levels=None
    unet_feat_mult=1; nb_unet_conv_per_level=1; int_steps=7; int_downsize=2
    bidir=False; use_probs=False; src_feats=1; trg_feats=1; unet_half_res=False;# unet_half_res=True
    in_channel = 45
    net = Unet(ConvBlock,inshape,infeats=in_channel*2,nb_features=nb_unet_features,nb_levels=nb_unet_levels,            feat_mult=unet_feat_mult,nb_conv_per_level=nb_unet_conv_per_level,half_res=unet_half_res,)
    net.remaining[2].main2 = nn.Conv3d(32,13,1)
    net.remaining[2].norm2 = nn.Identity()
    net.remaining[2].activation2 = nn.Tanh()

    _pool = [2] * net.nb_levels
    net.pooling = [nn.AvgPool3d(2) for s in _pool]
    net.upsampling = [nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False) for s in _pool]




    idx_test_all = torch.tensor([[0,1,15,16],[2,3,10,11],[4,5,17,18],[6,7,12,13],[8,9,14,19]])
    idx_test = idx_test_all[fold_nu]
    a_cat_b, counts = torch.cat([idx_test, torch.arange(28)]).unique(return_counts=True)
    idx_train = a_cat_b[torch.where(counts.eq(1))]
    print('fold',fold_nu,'train ids',idx_train)

    #net = torch.load('l2o_miccai22_vxadam_all_aug_fold'+str(fold_nu)+'_iter15000.pth')

    net.cuda()
    print(countParameters(net))


    net.cuda()
    scaler = torch.cuda.amp.GradScaler()


    t0 = time.time()
    net.train()
    run_loss = torch.zeros(24000)
    #run_loss[:15000] = net.run_loss
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,3000,0.5)
    for i in range(24000):
        idx = idx_train[torch.randperm(len(idx_train))[0:1]]#+1
        #if(i>400):
        #    idx = 255
        label_fix = mind_all_fix[idx].cuda()#torch.randn(1, 12, 80, 96, 80).cuda()#F.avg_pool2d(weight.view(1,-1,1,1).cuda()*F.one_hot(seg_all[pairs[idx,0]].cuda(),25).unsqueeze(0).float().permute(0,3,1,2),2)
        label_mov = mind_all_mov[idx].cuda()#torch.randn(1, 12, 80, 96, 80).cuda()#F.avg_pool2d(weight.view(1,-1,1,1).cuda()*F.one_hot(seg_all[pairs[idx,1]].cuda(),25).unsqueeze(0).float().permute(0,3,1,2),2)
        #mask_fix = (F.avg_pool3d(mask_all_fix[idx].cuda(),5,stride=2,padding=2)==1).float()

        N = len(keypts_all_fix[int(idx)])
        key_idx = torch.randperm(N)[:1024]
        keypts_fix = keypts_all_fix[int(idx)].cuda()[key_idx]
        keypts_mov = keypts_all_mov[int(idx)].cuda()[key_idx]
        pred_xyz = keypts_all_pred[int(idx)].cuda()[key_idx]
        alpha_tps = .1
        aug_i = int(torch.randint(5,(1,)))
        if(aug_i==1):
            pred_xyz = torch.zeros_like(pred_xyz)
        if(aug_i==2):
            pred_xyz = (pred_xyz+keypts_mov-keypts_fix)*.5
        if(aug_i==3):
            pred_xyz = (pred_xyz+keypts_mov-keypts_fix)*.5+0.2*torch.randn_like(keypts_fix)
            alpha_tps = 1
        if(aug_i==4):
            pred_xyz = pred_xyz*.5

        with torch.cuda.amp.autocast():
            ident = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H//2,W//2,D//2),align_corners=True).permute(0,4,1,2,3).cuda()
            with torch.no_grad():
                dense_flow_init = thin_plate_dense(keypts_fix.unsqueeze(0).cuda(), pred_xyz.unsqueeze(0).cuda(), (H//2, W//2, D//2), 4, alpha_tps)
                dense_flow_gt = thin_plate_dense(keypts_fix.unsqueeze(0).cuda(), (keypts_mov-keypts_fix).unsqueeze(0).cuda(), (H//2, W//2, D//2), 4, 0.1)
                pred_xyz_sample = F.grid_sample(dense_flow_gt.permute(0,4,1,2,3),keypts_fix.view(1,-1,1,1,3)).squeeze().t()#keypts_all_pred[int(idx)].cuda()



            init_lr = dense_flow_init.permute(0,4,1,2,3).detach().data#torch.randn_like(ident)*0.001
            init_lr.requires_grad = True
            field_lr = torch.zeros_like(init_lr)
            patch = field_lr.view(1,3,1,-1,1)+fixed_off
            sampled = F.grid_sample(label_mov,((init_lr+ident).view(1,3,1,-1,1)+patch).permute(0,2,3,4,1),align_corners=True)
            ssd = (sampled-label_fix.view(1,12,1,-1,1)).pow(2).mean(1)
            hidden_state = torch.zeros(1,10,H//2,W//2,D//2).cuda()
            scheduler.step()
            loss = 0
            for j in range(8):
                input = torch.cat((field_lr,patch.view(1,24,H//2,W//2,D//2),hidden_state,ssd.view(1,8,H//2,W//2,D//2)),1)
                output = checkpoint(net,F.pad(input,(4,4)))[...,4:-4]
                #output = checkpoint(net[9:],torch.cat((checkpoint(net[:9],input),input),1))
                field_lr = F.avg_pool3d(F.avg_pool3d(output[:,:3],3,stride=1,padding=1),3,stride=1,padding=1)
                with torch.no_grad():
                    patch = (field_lr).view(1,3,1,-1,1)+fixed_off
                    sampled = F.grid_sample(label_mov,((init_lr+ident).view(1,3,1,-1,1)+patch).permute(0,2,3,4,1),align_corners=True)
                    ssd = (sampled-label_fix.view(1,12,1,-1,1)).pow(2).mean(1)
                hidden_state = output[:,3:]
                field_lr_keypt = F.grid_sample(init_lr+field_lr,keypts_fix.view(1,-1,1,1,3)).squeeze().t()
                loss1 = ((keypts_mov-(keypts_fix+field_lr_keypt)).mul(100).pow(2).sum(-1).sqrt().mean())
                #deep-supervision
                loss += (j+1)*loss1

            #loss += (j+1)*nn.MSELoss()(field_lr_keypt,pred_xyz_sample*(j+1)/8)#field_lr*mask_fix,dense_flow_gt.permute(0,4,1,2,3)*mask_fix*(j+1)/8)
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        if(i%3==2):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()



        with torch.no_grad():
            loss1 = ((keypts_mov-(keypts_fix+field_lr_keypt)).mul(100).pow(2).sum(-1).sqrt().mean())
            run_loss[i] = loss1#loss.item()
            #output = net(output)
        if(i%150==49):
            t1 = time.time()
            gpu_usage()
            print(i,t1-t0,'sec',run_loss[i-20:i-1].mean())#,d_adam.mean())
        if(i%4500==1499):
            net.cpu()
            net.run_loss = run_loss
            torch.save(net,'learn2optimise_4dct/l2o_miccai22_vxadam_all_aug_fold'+str(fold_nu)+'_iter'+str(i+1)+'.pth')
            net.cuda()






