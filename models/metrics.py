import torch
import torch.nn.functional as F

def accuracy(pred, target):
    pred = F.softmax(pred, dim=-1)
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(target)

def top_k_accuracy(pred, target, k):
    pred = F.softmax(pred, dim=-1)
    correct = 0
    _, maxk = torch.topk(pred, k, dim = -1, sorted = False)
    _, y = torch.topk(y, k, dim=-1, sorted = False)
    correct += torch.eq(maxk, y).sum().detach().item()
    return correct
