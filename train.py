import json
import os
from matplotlib.style import available

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_postup import FSRCNN, SuperResolutor

from model_preup import SuperResolutorUpscaled, SRCNN, VDSR
from sample_dataset import UpscaledDataset, RegularDataset

model_args = {}
train_folder = ''
val_folder = ''
save_folder = ''
batch_size = 20
epochs = 200

writer = SummaryWriter(log_dir="runs/fsrcnn")

def psnr(out, original, max_val=1):
    mse = torch.mean((out - original) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_model():
    available_models = ["SRCNN", "VDSR", "FSRCNN"]
    if model_args["name"] not in available_models:
        print(f"The model {model_args['name']} is not (yet?) implemented.")
        print(f"Possible values: {available_models}")
        raise NotImplementedError
    
    if model_args["name"] == "SRCNN":
        model = SRCNN()
    elif model_args["name"] == "VDSR":
        model = VDSR(model_args["n_layers"])
    elif model_args["name"] == "FSRCNN":
        # model = FSRCNN()
        model = SuperResolutor()
    return model

def get_data(val = False):
    preup = ["SRCNN", "VDSR"]
    if model_args["name"] in preup:
        train = UpscaledDataset(train_folder).get_loader(batch_size, True)
        val = UpscaledDataset(val_folder).get_loader(batch_size, True)
    else:
        train = RegularDataset(train_folder).get_loader(batch_size, True)
        val = RegularDataset(val_folder).get_loader(batch_size, True)
    return train, val


def process():
    model = get_model()
    train, val = get_data()

    save_path = os.path.join(save_folder, "model.pth")

    try:
        model.load_state_dict(torch.load(save_path))
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    model = model.cuda()


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
                writer.add_images("input/output/target",torch.cat((interpolate(batch_downsized[:8],(64,64)), output_train[:8], batch_original[:8])),step)
                # writer.add_images("input",batch_downsized,step)
                # writer.add_images("output",output_train,step)
                # writer.add_images("target",batch_original,step)

        torch.save(model.state_dict(), save_path)


def read_args():
    global model_args
    global train_folder
    global val_folder
    global save_folder
    global batch_size
    global epochs
    
    args = {}
    with open("parameters.json", "r") as read_file:
        args = json.load(read_file)

    train_folder = args["training"]["train_folder"]
    val_folder = args["training"]["val_folder"]
    save_folder = args["training"]["save_folder"]
    batch_size = args["training"]["batch_size"]
    epochs = args["training"]["n_epochs"]
    os.makedirs(save_folder, exist_ok=True)
    model_args = args["model"]
    
if __name__ == "__main__":
    read_args()
    process()
