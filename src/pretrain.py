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
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


class get_preDataset:
    def __init__(self, pth):
        '''
            Args:
                pth: the path of the dataset
        '''
        self.data          = []
        self.model         = Body('../model/body_pose_model.pth')
        if os.path.exists(pth):
            ls             = os.listdir(pth)
            # read data
            for dir_ in ls:
                if "good" in dir_:
                    new_ls = os.listdir(pth + "/" + dir_)
                    with tqdm(total=len(new_ls)) as pbar:
                        pbar.set_description('Processing(%s):'%dir_)
                        for pic in new_ls:
                            candidate, subset  = self.model(cv2.imread(pth + "/" + dir_ + "/" + pic))
                            if subset.shape[0] == 0:
                                continue
                            self.data.append({"x": self.seperate(candidate, subset), "y": 0})
                            pbar.update(1)
                elif "bad" in dir_:
                    new_ls = os.listdir(pth + "/" + dir_)
                    with tqdm(total=len(new_ls)) as pbar:
                        pbar.set_description('Processing(%s):'%dir_)
                        for pic in new_ls:
                            candidate, subset  = self.model(cv2.imread(pth + "/" + dir_ + "/" + pic))
                            if subset.shape[0] == 0:
                                continue
                            self.data.append({"x": self.seperate(candidate, subset), "y": 1})
                            pbar.update(1)
            # shuffle the data
            np.random.shuffle(self.data)
            with open("data.py", "w") as f:
                f.write("data="+str(self.data))
        else:
            raise FileNotFoundError("Can't find the directory! Please check again!!")

    def seperate(self, candidate, subset):
        person = []
        #for n in range(len(subset)):
        for i in range(18):
            index = int(subset[0][i])
            if index == -1:
                person.append(np.array([-999, -999]))
            else:
                person.append(candidate[index][:2])
        return person

    
if __name__ == "__main__":
    data_pth    = "D:/github/tensorflowhub/tmp/Machine Learning/my-openpose/src/data"
    train_set   = get_preDataset(data_pth)
    