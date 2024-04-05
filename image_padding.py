import cv2
import numpy as np
import os

class ImagePadding:
    
    def __init__(self, image, Size: int):
        # self.img_path = img_path
        self.Size = Size
        self.image = image
        self.oriH, self.oriW, _ = self.image.shape
        self.numH = (self.oriH // self.Size) + 1
        self.numW = (self.oriW // self.Size) + 1
        self.newH = (self.numH) * self.Size
        self.padH = int((self.newH - self.oriH) / 2)
        self.newW = (self.numW) * self.Size
        self.padW = int((self.newW - self.oriW) / 2)
        self.new_image = self.generate_image()
    
    def generate_image(self):
        new_image = np.zeros((self.newH, self.newW, 3), dtype=np.uint8)
        new_image[
            self.padH:(self.oriH + self.padH), 
            self.padW:(self.oriW + self.padW), 
            0:3
        ] = self.image
        
        return new_image
    
    def save_image(self, image_folder_path):
        for h in range(self.numH):
            for w in range(self.numW):
                path = os.path.join(image_folder_path, f'test_h{int(h)}_w{int(w)}.jpg')
                h_start = h * self.Size
                w_start = w * self.Size
                img = self.new_image[h_start:(h_start+self.Size), w_start:(w_start+self.Size), ]
                cv2.imwrite(path, img)
        return True
    
    def save_annotation(self, label_folder_path):
        pass