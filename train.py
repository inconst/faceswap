import os
import argparse
import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import FaceDataset
from utils import set_global_seed, make_image_from_batches

from arcface.model import Backbone
from models.AEI_net import AEINet_generator, AEINet_discriminator

p = argparse.ArgumentParser(description='Face swapping AEINet training')
p.add_argument("--device", type=str, default='cuda', help='Device used (cpu or cuda)')
p.add_argument("--seed", type=int, default=0, help='Random seed')
p.add_argument("--num_epochs", type=int, default=1000, help='Number of training epochs')
p.add_argument("--pretrained_dir", type=str, default='',
               help='Directory to load model from (if empty then training from scratch)')
p.add_argument("--display_freq", type=int, default=10, help='How often display training results (in iterations)')
p.add_argument("--dataset_path", type=str, default='./data/thumbnails128x128', help='Path to dataset directory')
p.add_argument("--dataset_size", type=int, default=70000, help='Number of images to take for training')
p.add_argument("--batch_size", type=int, default=16, help='Batch size')
args = p.parse_args()
set_global_seed(args.seed)
device = torch.device(args.device)

dataset = FaceDataset(args)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

arcface_model = Backbone(50, 0.6, 'ir_se').to(device)
arcface_model.eval()
arcface_model.load_state_dict(torch.load('saved_models/model_ir_se50.pth', map_location=device), strict=False)

model_G = AEINet_generator(c_id=512).to(device)
model_D = AEINet_discriminator().to(device)
if args.pretrained_dir:
    model_G.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'G_latest.pth'), map_location=device))
    model_D.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'D_latest.pth'), map_location=device))

opt_G = optim.Adam(model_G.parameters(), lr=4e-4, betas=(0, 0.999))
opt_D = optim.Adam(model_D.parameters(), lr=4e-4, betas=(0, 0.999))


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1 - X).mean()
    else:
        return torch.relu(X + 1).mean()


print('===> Started training')
stats_writer = SummaryWriter('tensorboard_stats')
total_iter = 0
for epoch in range(args.num_epochs):

    start_time = time.time()
    for iteration, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        Xs, Xt, same_person = data
        num_same_persons = same_person.sum().numpy()
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        # --------------

        with torch.no_grad():
            source_id = arcface_model(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))

        opt_G.zero_grad()
        Y, target_attributes = model_G(Xt, source_id)
        fake_output = model_D(Y)
        generated_attributes = model_G.get_attr(Y)
        generated_id = arcface_model(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=True))

        loss_adv = hinge_loss(fake_output, True)
        loss_id = (1 - torch.cosine_similarity(source_id, generated_id, dim=1)).mean()
        loss_attr = 0.0
        for i in range(len(target_attributes)):
            loss_attr += torch.pow(target_attributes[i] - generated_attributes[i], 2).mean()
        loss_attr /= 2.0
        if num_same_persons > 0:
            loss_rec = torch.sum(0.5 * torch.pow(Y - Xt, 2).reshape(Xt.shape[0], -1).mean(dim=1)
                                 * same_person) / same_person.sum()
        else:
            loss_rec = torch.tensor(0.0).to(device)
        total_loss_G = 1 * loss_adv + 10 * loss_attr + 1 * loss_id + 10 * loss_rec

        total_loss_G.backward()
        opt_G.step()

        # --------------

        opt_D.zero_grad()

        fake_output = model_D(Y.detach())
        real_output = model_D(Xs)
        loss_true = hinge_loss(fake_output, False)
        loss_fake = hinge_loss(real_output, True)
        total_loss_D = 0.5 * (loss_true + loss_fake)

        total_loss_D.backward()
        opt_D.step()

        # --------------

        # visualization of results
        if total_iter % args.display_freq == 0:
            image = make_image_from_batches(Xs, Xt, Y, num_images=6)
            image = np.transpose(image, (1, 2, 0))

            cv2.imshow("Result", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()

        stats_writer.add_scalars('last_run', {'loss_adv': loss_adv.item(), 'loss_id': loss_id.item(),
                                              'loss_attr': loss_attr.item(), 'loss_rec': loss_rec.item(),
                                              'loss_G': total_loss_G.item(), 'loss_D': total_loss_D.item()}, total_iter)

        total_iter += 1

    torch.save(model_G.state_dict(), 'saved_models/G_latest.pth')
    torch.save(model_D.state_dict(), 'saved_models/D_latest.pth')
    print("Finished epoch #{}. Time elapsed: {:.2f}s".format(epoch + 1, time.time() - start_time))

stats_writer.close()
