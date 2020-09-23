
import os
import argparse
import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from dataset import FaceDataset
from utils import set_global_seed, make_image_from_batches

from arcface.model import Backbone


p = argparse.ArgumentParser(description='SR training script')
p.add_argument("--result_dir", type=str, default='./results')
p.add_argument("--device", type=str, default='cuda')
p.add_argument("--seed", type=int, default=0, help='')
p.add_argument("--num_epochs", type=int, default=1000, help='Number of training epochs')
p.add_argument("--display_freq", type=int, default=50, help='')

p.add_argument("--dataset_path", type=str, default='./data/FFHQ_LR', help='Path to dataset directory')
p.add_argument("--dataset_size", type=int, default=10, help='')
p.add_argument("--batch_size", type=int, default=16, help='')

args = p.parse_args()
os.makedirs(args.result_dir, exist_ok=True)
set_global_seed(args.seed)
device = torch.device(args.device)

dataset = FaceDataset(args)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

arcface_model = Backbone(50, 0.6, 'ir_se').to(device)
arcface_model.eval()
arcface_model.load_state_dict(torch.load('saved_models/model_ir_se50.pth', map_location=device), strict=False)

print('===> Started training. Results are in {}'.format(args.result_dir))
total_iter = 0
for epoch in range(args.num_epochs):

    start_time = time.time()
    for iteration, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        Xs, Xt, same = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)

        res = arcface_model(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))

        # visualization of results
        if total_iter % args.display_freq == 0:
            image = make_image_from_batches(Xs, Xt, Xs, num_images=6)
            image = np.transpose(image, (1, 2, 0))

            cv2.imshow("Result", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()

        total_iter += 1

    print("Finished epoch #{}. Time elapsed: {:.2f}s".format(epoch + 1, time.time() - start_time))


