import collections
import numpy as np
import torch
import model
import time
import util
import cv2
import os

from matplotlib import pyplot as plt
from torch import nn
from body import Body
from torch.cuda import amp
from transformers import get_cosine_schedule_with_warmup
from pretrain import get_preDataset


# class SittingPoseNet(nn.Module):
#     def __init__(self):
#         super(SittingPoseNet, self).__init__()
#         self.model             = Body('../model/body_pose_model.pth')
#         self.relu              = nn.ReLU()
#         self.sigmoid           = nn.Sigmoid()
#         self.dense1            = nn.Linear(2*18, 15)
#         self.dense2            = nn.Linear(12, 1)
    
#     def forward(self, frame):
#         with amp.autocast():
#             candidate, subset  = self.model(frame)
#             candidate          = candidate[:, :2]
#             subset             = subset[:, :18]
#             tmp                = []
#             for person in range(len(subset)):
#                 person_points  = []
#                 idx            = subset[person]
#                 person_points.append(candidate[idx])
#                 person_points  = torch.nn.Flatten(person_points)
#                 out1           = self.dense1(person_points)
#                 out1           = self.relu(out1)
#                 out2           = self.dense2(out1)
#                 out2           = self.sigmoid(out2)
#                 tmp.append(out2)
#         return tmp

    
###############################################################
def transfer_dict(dic):
    tmp = collections.OrderedDict()
    for i in dic.keys():
        tmp[".".join(i.split(".")[1:])] = dic[i]
    return tmp


class SittingPoseNet(nn.Module):
    def __init__(self):
        super(SittingPoseNet, self).__init__()
        self.relu              = nn.ReLU()
        self.sigmoid           = nn.Sigmoid()
        self.dense1            = nn.Linear(36, 32)
        self.dense2            = nn.Linear(32, 12)
        self.dense3            = nn.Linear(12, 2)
    
    def forward(self, candidate):
        with amp.autocast():
            candidate      = candidate.float()
            person_points  = torch.nn.Flatten()(candidate)
            out1           = self.dense1(person_points)
            out1           = self.relu(out1)
            out2           = self.dense2(out1)
            out2           = self.relu(out2)
            out3           = self.dense3(out2)
           # out3           = self.sigmoid(out3)
        return out3
###############################################################
    

# class get_Dataset(torch.utils.data.Dataset):
#     def __init__(self, pth):
#         '''
#             Args:
#                 pth: the path of the dataset
#         '''
#         self.data          = []
#         if os.path.exists(pth):
#             ls             = os.listdir(pth)
#             # read data
#             for dir_ in ls:
#                 if "good" in dir_:
#                     new_ls = os.listdir(pth + "/" + dir_)
#                     for pic in new_ls:
#                         self.data.append({"x": cv2.imread(pth + "/" + dir_ + "/" + pic), "y": 0})
#                 elif "bad" in dir_:
#                     new_ls = os.listdir(pth + "/" + dir_)
#                     for pic in new_ls:
#                         self.data.append({"x": cv2.imread(pth + "/" + dir_ + "/" + pic), "y": 1})
#             # shuffle the data
#             np.random.shuffle(self.data)
#         else:
#             raise FileNotFoundError("Can't find the directory! Please check again!!")
            
        
#     def __len__(self):
#         return len(self.data[1])
    
#     def __getitem__(self, idx):
#         '''
#             This function will return the image_frames and labels
#         '''
#         frames, labels     = self.data[idx]["x"], self.data[idx]["y"]
#         return frames, labels  # torch.tensor(frames), torch.tensor(labels)

###############################################################################
class get_Dataset(torch.utils.data.Dataset):
    def __init__(self, pth):
        '''
            Args:
                pth: the path of the dataset
        '''
        #self.data           = get_preDataset(pth).data
        #torch.save(self.data, "pretrain.data")
        self.data = torch.load("pretrain.data")
            
    def __len__(self):
        return len(self.data[1])
    
    def __getitem__(self, idx):
        '''
            This function will return the image_frames and labels
        '''
        points, labels     = self.data[idx]["x"], self.data[idx]["y"]
        return torch.tensor(points), torch.tensor(labels)
#################################################################################d 
    
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device(i):
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
    
    
# def train_with_amp(net, train_set, criterion, epochs, batch_size,
#                    gradient_accumulate_step, max_grad_norm, finetuning = False, 
#                    lr = .001, optimizer = "Adam"):
#     net.train()   
#     # instantiate a scalar object 
#     ls             = []
#     num_gpu        = torch.cuda.device_count()
#     device_ids     = [try_gpu(i) for i in range(num_gpu)]
#     device         = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print("\ntrain on %s\n"%str(device_ids))
    
#     if type(finetuning) != bool:
#         cnt        = 0
#         # ensure the net layer number is positive
#         if finetuning < 0:
#             finetuning += len(list(net.children()))
#         for k in net.children():
#             cnt   += 1
#             if not cnt > finetuning:
#                 for param in k.parameters():
#                     param.requires_grad = False  # freeze the params
    
#     # initialize the optimizer
#     if optimizer.lower() == "adam":
#         optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
#     elif optimizer.lower() == "adamw":
#         optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
#     elif optimizer.lower() == "sgd":
#         optimizer  = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
#     else:
#         print("Don't support this optimizer function by far!")
#         return 0
    
#     # initialize the scheduler
#     scheduler   = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 0, 
#                                     num_training_steps = len(torch.utils.data.DataLoader(train_set, batch_size = batch_size)), 
#                                     num_cycles = 0.5)
    
#     # initialize other things
#     enable_amp  = True if "cuda" in device_ids[0].type else False
#     scaler      = amp.GradScaler(enabled = enable_amp)
#     net         = nn.DataParallel(net, device_ids = device_ids)
#     net.to(device)
#     train_iter  = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
#     for epoch in range(epochs):
#         for idx, value in enumerate(train_iter):
#             ini_time       = time.time()
#             frames, labels = value
#             frames         = frames.to(device_ids[0])
#             labels         = labels.to(device_ids[0])
#             # when forward process, use amp
#             with amp.autocast(enabled = enable_amp):
#                 output     = net(frames)  
#             loss           = criterion(output, labels.view(-1,1).float())

#             # prevent gradient to be 0
#             if gradient_accumulate_step > 1:
#                 # if RAM is not enough, we use "gradient_accumulate" to solve it
#                 loss      = loss / gradient_accumulate_step
#             print(loss)
#             # avoid the gradient vanishing
#             scaler.scale(loss).mean().backward()

#             # do the gradient clip
#             gradient_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
#             if (idx + 1) % gradient_accumulate_step == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
#                 scheduler.step()
#             # print loss every 100 iterations
#             if idx % 100 == 0 or idx == len(train_iter) -1:
#                 with torch.no_grad():
#                     print("==============Epochs "+ str(epoch) + " ======================")
#                     print("loss: " + str(loss) + "; grad_norm: " + str(gradient_norm))
#                 ls.append(loss.item())
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': net.state_dict(),
#                     'param_groups': optimizer.state_dict()["param_groups"],
#                     'loss': ls
#                 },"./checkpoint.params")
#             with open("train_log", "a") as f:
#                 f.write("Epoch %s, Batch %s: %.4f sec\n"%(epoch, idx, time.time() - ini_time))

def train_with_amp(net, train_set, criterion, epochs, batch_size,
                   gradient_accumulate_step, max_grad_norm, finetuning = False, 
                   lr = .001, optimizer = "Adam"):
    net.train()   
    # instantiate a scalar object 
    ls             = []
    num_gpu        = torch.cuda.device_count()
    device_ids     = [try_gpu(i) for i in range(num_gpu)]
    device         = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("\ntrain on %s\n"%str(device_ids))
    
    if type(finetuning) != bool:
        cnt        = 0
        # ensure the net layer number is positive
        if finetuning < 0:
            finetuning += len(list(net.children()))
        for k in net.children():
            cnt   += 1
            if not cnt > finetuning:
                for param in k.parameters():
                    param.requires_grad = False  # freeze the params
    
    # initialize the optimizer
    if optimizer.lower() == "adam":
        optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
    elif optimizer.lower() == "adamw":
        optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
    elif optimizer.lower() == "sgd":
        optimizer  = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
    else:
        print("Don't support this optimizer function by far!")
        return 0
    
    # initialize the scheduler
    scheduler   = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 0, 
                                    num_training_steps = len(torch.utils.data.DataLoader(train_set, batch_size = batch_size)), 
                                    num_cycles = 0.5)
    
    # initialize other things
    enable_amp  = True if "cuda" in device_ids[0].type else False
    scaler      = amp.GradScaler(enabled = enable_amp)
    # net         = nn.DataParallel(net, device_ids = device_ids)
    net.to(device)
    train_iter  = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    for epoch in range(epochs):
        for idx, value in enumerate(train_iter):
            ini_time                  = time.time()
            candidate, labels         = value
            candidate                 = candidate.to(device_ids[0])
            labels                    = labels.to(device_ids[0])
            # when forward process, use amp
            with amp.autocast(enabled = enable_amp):
                output                = net(candidate)  
            # print(output)
            loss                      = criterion(output, labels.view(-1))#.float())

            # prevent gradient to be 0
            if gradient_accumulate_step > 1:
                # if RAM is not enough, we use "gradient_accumulate" to solve it
                loss                  = loss / gradient_accumulate_step
            # print(loss)
            # avoid the gradient vanishing
            scaler.scale(loss).mean().backward()

            # do the gradient clip
            gradient_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            if (idx + 1) % gradient_accumulate_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            # print loss every 100 iterations
            if idx % 100 == 0 or idx == len(train_iter) -1:
                with torch.no_grad():
                    print("==============Epochs "+ str(epoch) + " ======================")
                    print("loss: " + str(loss) + "; grad_norm: " + str(gradient_norm))
                ls.append(loss.item())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'param_groups': optimizer.state_dict()["param_groups"],
                    'loss': ls
                },"./checkpoint.params")
            with open("train_log", "a") as f:
                f.write("Epoch %s, Batch %s: %.4f sec\n"%(epoch, idx, time.time() - ini_time))
    
    plt.figure(figsize=(10,6))
    plt.plot(np.array(ls), alpha=.8)
    plt.show()