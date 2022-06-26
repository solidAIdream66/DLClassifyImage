import argparse
from models import create_model, save_checkpoint
import torch
from data_load import load_categories, load_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Train a new network on a data set to classify flowers'
    )
    parser.add_argument('data_dir', help='data directory with train, validation, test dataset')
    parser.add_argument('--save_dir', default='saved_models',
                        help='set directory to save checkpoints')
    parser.add_argument('--arch', default='vgg11', help='model architecture in torchvision.models')
    parser.add_argument('--lr', type=float, default=0.001, help='set hyperparameters: learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='set hyperparameters: hidden units')
    parser.add_argument('--epochs', type=int, default=2, help='set hyperparameters: epochs')
    parser.add_argument('--category_name', default='cat_to_name.json',
                        help='Use a mapping of categories to real names')
    args = parser.parse_args()
    print('parse arguments:\n\t', args, '\n')

    # load data
    data_dir = args.data_dir
    trainloaders, validloaders, testloaders = load_data(data_dir)

    # model
    arch = args.arch
    lr = args.lr
    hidden_units = args.hidden_units
    category_name = args.category_name
    cat_to_name = load_categories(category_name)
    output_units= len(cat_to_name)
    model, criterion, optimizer = create_model(arch, lr, hidden_units, output_units)
    print('model:\n\t', model, '\n')

    # train classifier, track train loss, valid loss, and accuracy
    epochs = args.epochs
    running_loss = 0
    train_losses = []
    valid_losses = []
    print_every = 10
    track_per_epochs = int(len(trainloaders)/print_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for e in range(epochs):

        for i, (images, labels) in enumerate(trainloaders):
            images = images.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloaders:
                        images = images.to(device)
                        labels = labels.to(device)

                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/print_every)
                valid_losses.append(valid_loss/len(validloaders))

                print(f"Iterator: {int((i+1)/print_every)+e*track_per_epochs}/{epochs*track_per_epochs}..",
                      f"Train loss: {running_loss/print_every:.2f}..",
                      f"Valid loss: {valid_loss/len(validloaders):.2f}..",
                      f"Accuracy: {accuracy/len(validloaders):.2f}..")
                running_loss = 0

    # test network
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloaders:
            images = images.to(device)
            labels = labels.to(device)

            logps = model.forward(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"\nTest loss: {test_loss/len(testloaders):.2f}..",
          f"Accuracy: {accuracy/len(testloaders):.2f}..")


    # save the checkpoints
    save_dir = args.save_dir
    save_checkpoint(model, arch, lr, hidden_units, output_units, save_dir, accuracy/len(testloaders))

    plt.plot(train_losses[:], label='train loss')
    plt.plot(valid_losses[:], label='valid loss')
    plt.legend(frameon=False)
    plt.show()
