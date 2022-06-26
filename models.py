import torch
from torch import nn, optim
import torchvision

def create_model(arch, lr, hidden_units, output_units):
    criterion = nn.NLLLoss()
    model = torchvision.models.vgg11(pretrained=True)

    # freeze
    for params in model.parameters():
        params.requires_grad = False

    # classifier
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(hidden_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(hidden_units, output_units),
                                     nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    hidden_units = checkpoint['hidden_units']
    output_units = checkpoint['output_units']
    model, criterion, optimizer = create_model(arch, lr, hidden_units, output_units)
    model.classifier.load_state_dict(checkpoint['state_dict'])
    return model, criterion, optimizer


def save_checkpoint(model, arch, lr, hidden_units, output_units, save_dir, accuracy):
    checkpoint = {'arch': arch,
                  'lr': lr,
                  'hidden_units': hidden_units,
                  'output_units': output_units,
                  'state_dict': model.classifier.state_dict()}
    checkpoint_path = "{}/checkpoint_{}.pth".format(save_dir, int(accuracy*100))
    torch.save(checkpoint, checkpoint_path)
