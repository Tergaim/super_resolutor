import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SuperResolutor
from sample_dataset import SampleDataset

args = {}
train_folder = ''
save_folder = ''
batch_size = 20
epochs = 200

writer = SummaryWriter(log_dir="runs/13_1_7")

def psnr(out, original, max_val=1):
    mse = torch.mean((out - original) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def process():
    model = SuperResolutor()
    save_path = os.path.join(save_folder, "model.pth")

    try:
        model.load_state_dict(torch.load(save_path))
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    model = model.cuda()

    train = SampleDataset(train_folder).get_loader(batch_size, True)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Training...')
    step = 0

    loss_agreg = 0
    for epoch in tqdm(range(epochs)):
        for batch_downsized, batch_original in tqdm(train):
            batch_downsized = batch_downsized.cuda()
            batch_original = batch_original.cuda()
            step += 1
            # compute
            optimizer.zero_grad()
            output_train = model(batch_downsized)
            # print(output_train.shape)
            # print(batch_original.shape)
            loss = loss_func(output_train, batch_original)
            loss.backward()
            loss_agreg += loss.item()
            optimizer.step()

            # training logs
            if (step % 20 == 0):
                writer.add_scalar('MSELoss/train', loss_agreg, step)
                loss_agreg = 0
                psnr_train = psnr(output_train, batch_original)
                writer.add_scalar('PSNR/train', psnr_train, step)
            
            if (step %300 == 0):
                writer.add_images("input/output/target",torch.cat((batch_downsized[:8], output_train[:8], batch_original[:8])),step)
                # writer.add_images("input",batch_downsized,step)
                # writer.add_images("output",output_train,step)
                # writer.add_images("target",batch_original,step)

        torch.save(model.state_dict(), save_path)


def read_args():
    global args
    global train_folder
    global save_path
    global batch_size
    global epochs
    
    with open("parameters.json", "r") as read_file:
        args = json.load(read_file)["training"]

    train_folder = args["train_folder"]
    save_folder = args["save_folder"]
    batch_size = args["batch_size"]
    epochs = args["n_epochs"]
    os.makedirs(save_folder, exist_ok=True)
    

if __name__ == "__main__":
    read_args()
    process()
