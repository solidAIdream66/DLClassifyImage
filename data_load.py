from torchvision import transforms, datasets
import torch
import json


def load_data(data_dir):
    # dir
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transforms
    train_transform = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    other_transform = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    # datasets
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=other_transform)
    test_datasets = datasets.ImageFolder(test_dir, transform=other_transform)

    # dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=False)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False)

    return trainloaders, validloaders, testloaders, train_datasets.class_to_idx


def load_categories(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
