from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
import random
import os


class FaceDataset(TensorDataset):
    def __init__(self, args):
        self.image_paths = glob.glob(f'{args.dataset_path}/*.*g')[:args.dataset_size]
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.same_face_probability = 0.2

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        Xs = self.load_image(image_path)
        transformed_Xs = self.transforms(Xs)

        if random.random() > self.same_face_probability:
            image_path = random.choice(self.image_paths)
            Xt = self.load_image(image_path)
            same_person = False
        else:
            Xt = Xs.copy()
            same_person = True
        transformed_Xt = self.transforms(Xt)
        return transformed_Xs, transformed_Xt, same_person

    def load_image(self, path):
        image = cv2.imread(path)
        image = Image.fromarray(image)
        return image

    def __len__(self):
        return len(self.image_paths)