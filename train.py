import os
import torch
import wandb
import numpy as np
import time
import copy

from torch import nn, optim
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

from load_data import load_data
from test import eval
from config import *
import models

def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, scheduler, epoch, device):
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    count = 0
    patience = 20
    checkpoint = {}
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epoch):
            print(f"------------------------------EPOCH {e}------------------------------")
            model.train()
            progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
            losses = 0
            correct = 0
            size = 0
            for batch, (X, y) in progress:
                X = X.to(device); y= y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.detach().item()
                size += len(X)
                
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                train_loss = losses/(batch+1)
                train_acc = correct/size
                progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))

                # for name, param in model.named_parameters():
                #     if 'weight' in name:
                #         param.data = torch.clamp(param.data, 0, 1)

            valid_acc, valid_loss, _ = eval(valid_dataloader, model, loss_fn, False, device = device)
            print(f"Test Loss: {valid_loss}, Test Accuracy: {valid_acc}")
            if scheduler:
                scheduler.step(valid_loss)

            metrics = {
                "train/loss": train_loss,
                "train/epoch": e,
                "train/accuracy": train_acc,
                "train/learnrate": optimizer.param_groups[0]['lr'],
                "valid/loss": valid_loss,
                "valid/accuracy": valid_acc
            }
            wandb.log(metrics, step=e)

            #early stopping
            if valid_acc < best_valid_acc:
                count += 1
                # if count >= patience:
                #     break
            else:
                count = 0
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                cur_train_loss = train_loss
                cur_train_acc = train_acc
                print(f'best epoch: {e}')
                checkpoint['model_weights'] = model.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['scheduler'] = scheduler.state_dict()
                    
    return cur_train_loss, cur_train_acc, best_valid_loss, best_valid_acc, checkpoint

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=project,

    name = name,

    notes = description,
    
    tags = tags,

    group = group,
    
    # track hyperparameters and run metadata
    config=config
)

train_dataloader, test_dataloader = load_data(dataset=config['dataset'], root=config['root'], batch_size=config['batch_size'], input_size=config['input_shape'])

model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
model = model.to(config['device'])
print(model)
summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']))

loss_fn = getattr(nn, config['loss_fn'])()
optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), lr=config['lr'], **dict(config['optimizer']['args']))
scheduler = getattr(optim.lr_scheduler, config['lr_scheduler']['name'])(optimizer, **dict(config['lr_scheduler']['args']))

shutil.copyfile(f'./models/{config["model"]["name"]}.py', f'{save_dir}/{config["model"]["name"]}.py')

# wandb.watch(model, loss_fn, log="all", log_freq=1)
train_loss, train_acc, valid_loss, valid_acc, checkpoint = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, config['epoch'], device = config['device'])
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))
print("Valid: \n\tAccuracy: {}, Avg loss: {} \n".format(valid_acc, valid_loss))

# Test model
model.load_state_dict(checkpoint['model_weights'])
test_acc, test_loss, test_table = eval(test_dataloader, model, loss_fn, device = config['device'])
print("Test: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['train_accuracy'] = train_acc
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['test_accuracy'] = test_acc
wandb.summary['test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})
print(f'checkpoint keys: {checkpoint.keys()}')

torch.save(checkpoint, f'{config["save_dir"]}/{config["model"]["name"]}_epochs{config["epoch"]}.pth')
# m = torch.jit.script(model)
# script = torch.jit.save(m, f'{save_dir}/RGB_Plan_v10_epochs{epoch}_entire_model.pth')
art = wandb.Artifact(f'{config["model"]["name"]}_{config["dataset"]}', type="model")
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}_epochs{config["epoch"]}.pth')
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}.py')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()