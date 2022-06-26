import torch
from torch import nn, optim
import torchvision

def create_model(arch, lr, hidden_units, output_units, class_to_idx):
    resnet18 = torchvision.models.resnet18(pretrained=True)
    alexnet = torchvision.models.alexnet(pretrained=True)
    vgg16 = torchvision.models.vgg16(pretrained=True)
    models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    model = models[arch]

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
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    hidden_units = checkpoint['hidden_units']
    output_units = checkpoint['output_units']
    class_to_idx = checkpoint['class_to_idx']
    model, criterion, optimizer = create_model(arch, lr, hidden_units, output_units, class_to_idx)
    model.classifier.load_state_dict(checkpoint['state_dict'])
    return model, criterion, optimizer


def save_checkpoint(model, arch, lr, hidden_units, output_units, save_dir, accuracy):
    checkpoint = {'arch': arch,
                  'lr': lr,
                  'hidden_units': hidden_units,
                  'output_units': output_units,
                  'state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx}
    checkpoint_path = "{}/checkpoint_{}.pth".format(save_dir, int(accuracy*100))
    torch.save(checkpoint, checkpoint_path)
