import torch
import model

def load_model(path):
    net=torch.load(path)
    return net