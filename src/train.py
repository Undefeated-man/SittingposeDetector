import torch
import util

from torch import nn
from transformers import get_cosine_schedule_with_warmup
from sit import SittingPoseNet, train_with_amp, get_Dataset


if __name__ == "__main__":
    model       = SittingPoseNet()
    data_pth    = "D:/github/tensorflowhub/tmp/Machine Learning/my-openpose/src/data"
    loss        = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss() # nn.functional.binary_cross_entropy  # 
    batch_size  = 128
    epoch       = 50
    lr          = 0.001
    train_set   = get_Dataset(data_pth)
    # print(torch.load("checkpoint.params")["model_state_dict"])
    #model.load_state_dict(torch.load("checkpoint.params")["model_state_dict"])
    train_with_amp(model, train_set, loss, epoch, batch_size, 1, 1000, False, lr, "adamw")