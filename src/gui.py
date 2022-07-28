import os
import sys
import time

from threading import Thread
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Masker(QThread):
    """
        Func:
            Using the 'mask' file to pass the signal, if the signal is True, we mask everything; otherwise, we take the mask away.
    """
    def __init__(self):
        super().__init__()
        
        self.mask   = [1, 1]
        self.flag   = True

        # Create the main window
        self.window = QMainWindow()

        self.window.setWindowOpacity(0.7)
        self.window.setAttribute(Qt.WA_NoSystemBackground, True)
        self.window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.window.showFullScreen()
        self.window.showMinimized()
        
    def blurry(self):
        if self.flag:
            if self.mask[0] and self.mask[0] != self.mask[1]:
                self.window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)   # Stay over any other apps
                self.window.showMaximized()
                self.mask[1] = self.mask[0]
            elif ~self.mask[0] and self.mask[0] != self.mask[1]:
                self.window.setWindowFlags(Qt.FramelessWindowHint | Qt.Widget)                 # Stay under any other apps
                self.window.showMinimized()
                self.mask[1] = self.mask[0]
            
            if not os.path.exists("mask"):
                with open("mask", "w") as f:
                    f.write("F")

            with open("mask", "r") as f:
                tmp = f.read()
                if tmp == "T":
                    self.mask[0] = 1
                elif tmp == "F":
                    self.mask[0] = 0
    
    def kill_all(self):
        """
            Func:
                Deciding whether kill all the process
        """
        if self.flag:
            if not os.path.exists("flag"):
                with open("flag", "w") as f:
                    f.write("T")

            with open("flag", "r") as f:
                tmp = f.read()
                if tmp == "F":
                    self.flag = False
        else:
            sys.exit(0)
    
    def start(self):
        self.timer1 = QTimer(self)
        self.timer2 = QTimer(self)
        self.timer1.timeout.connect(self.blurry)
        self.timer2.timeout.connect(self.kill_all)
        self.timer1.start(100)
        self.timer2.start(100)

                    
if __name__ == '__main__':
    app    = QApplication(sys.argv)
    m      = Masker()
    m.start()
    sys.exit(app.exec_())