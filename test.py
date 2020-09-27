import sys

sys.path.append('./arcface/')
import os
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import ImageTk
import PIL.Image

from tkinter import *
from tkinter import filedialog

from arcface.model import Backbone
from arcface.mtcnn import *
from models.AEI_net import AEINet_generator

p = argparse.ArgumentParser(description='SR training script')
p.add_argument("--pretrained_dir", type=str, default='saved_models/',
               help='Directory to load model from')
p.add_argument("--device", type=str, default='cpu', help='Device used (cpu or cuda)')
args = p.parse_args()


class App(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Face swap')

        self.input_size = (128, 128)
        device = torch.device(args.device)
        self.device = device
        self.detector = MTCNN(device)

        self.arcface_model = Backbone(50, 0.6, 'ir_se').to(device)
        self.arcface_model.eval()
        self.arcface_model.load_state_dict(torch.load('saved_models/model_ir_se50.pth', map_location=device),
                                           strict=False)

        self.model_G = AEINet_generator(c_id=512).to(device)
        self.model_G.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'G_latest.pth'), map_location=device))
        self.model_G.eval()

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.source_image_data = []
        self.target_image_data = []

        self.black_image = ImageTk.PhotoImage(PIL.Image.fromarray(np.zeros((*self.input_size, 3), dtype=np.uint8)))
        self.source_image_label = Label(image=self.black_image)
        self.source_image_label.grid(row=0, column=0)
        self.target_image_label = Label(image=self.black_image)
        self.target_image_label.grid(row=0, column=1)
        self.result_image_label = Label(image=self.black_image)
        self.result_image_label.grid(row=0, column=2)

        Button(text="Pick source", command=self.open_source).grid(row=1, column=0)
        Button(text="Pick target", command=self.open_target).grid(row=1, column=1)
        Button(text="Swap faces", command=self.swap_faces).grid(row=1, column=2)

    def open_source(self):
        filename = filedialog.askopenfilename()
        if filename != "":
            self.source_image_data = self.load_image(filename)
            self.source_image_TK = ImageTk.PhotoImage(self.source_image_data[0])
            self.source_image_label.config(image=self.source_image_TK,
                                           width=self.source_image_TK.width(), height=self.source_image_TK.height())

    def open_target(self):
        filename = filedialog.askopenfilename()
        if filename != "":
            self.target_image_data = self.load_image(filename)
            self.target_image_TK = ImageTk.PhotoImage(self.target_image_data[0])
            self.target_image_label.config(image=self.target_image_TK,
                                           width=self.target_image_TK.width(), height=self.target_image_TK.height())

    def load_image(self, filename):
        image_bgr = PIL.Image.fromarray(cv2.imread(filename))
        if image_bgr.width != self.input_size[0] or image_bgr.height != self.input_size[1]:
            image_bgr = self.detector.align(image_bgr, crop_size=self.input_size)
        image_tensor = self.test_transform(image_bgr).unsqueeze(0).to(self.device)
        image_rgb = PIL.Image.fromarray(np.array(image_bgr)[:, :, ::-1])
        return image_rgb, image_tensor

    def swap_faces(self):
        if len(self.source_image_data) == 0 or len(self.target_image_data) == 0:
            print('Pick both source and target images first!!!')
            return

        _, Xs = self.source_image_data
        _, Xt = self.target_image_data
        with torch.no_grad():
            embeds = self.arcface_model(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
            Yt, _ = self.model_G(Xt, embeds)
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
            Yt_rgb = Yt[:, :, ::-1]
            Yt_rgb = np.clip(Yt_rgb * 255.0, 0.0, 255.0).astype(np.uint8)

            self.result_image_TK = ImageTk.PhotoImage(PIL.Image.fromarray(Yt_rgb))
            self.result_image_label.config(image=self.result_image_TK,
                                           width=self.result_image_TK.width(), height=self.result_image_TK.height())


if __name__ == "__main__":
    app = App()
    app.mainloop()
