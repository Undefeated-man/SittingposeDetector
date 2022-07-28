import os
import cv2
import copy
import torch
import numpy as np

from src import util
from src.body import Body
from threading import Thread
from time import sleep


class Main:
    """
        Func:
            Using multi-threadings to speed up the detecting process
        
        Args:
            net: the pretrained key-points-detecting net
    """
    def __init__(self, net):
        self.canvas  = None
        self.cap     = cv2.VideoCapture(0)
        self.flag    = True
        self.net     = net
        self.cap.set(3, 480)    # set the width
        self.cap.set(4, 480)    # set the height
        
    def catch(self):
        while self.cap and self.flag:
            ret, oriImg = self.cap.read()
            
            print(oriImg.shape)
            self.canvas = copy.deepcopy(oriImg)

    def detect(self):
        while self.flag:
            tmp = self.canvas
            if tmp is None:
                sleep(0.2)
                continue
            candidate, subset = self.net(tmp)
            canvas = util.draw_bodypose(tmp, candidate, subset)
            cv2.imshow('Good job! Remember to give me an A score!', canvas)       # create a window to show the image
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def stop_catching(self):
        flag_file = "flag"
        
        if not os.path.exists(flag_file):
            with open(flag_file, "w") as f:
                f.write("T")
        
        with open(flag_file, "r") as f:
            tmp = f.read()
        if tmp == "F":
            with open(flag_file, "w") as f:
                f.write("T")
        
        while self.flag:
            with open(flag_file, "r") as f:
                tmp = f.read()
                if tmp == "F":
                    self.flag = False
        
    def start(self):
        Thread(target=self.catch, daemon=True).start()
        Thread(target=self.detect).start()
        Thread(target=self.stop_catching, daemon=True).start()


if __name__ == "__main__":
    body_estimation = Body('model/body_pose_model.pth')
    # hand_estimation = Hand('model/hand_pose_model.pth')

    print(f"Torch device: {torch.cuda.get_device_name()}")
    main = Main(body_estimation)
    main.start()