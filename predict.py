import argparse
from models import load_checkpoint
from data_load import load_categories
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # scale
    width, height = image.size
    if width > height:
        new_height = 256
        new_width = int(width / height * 256)
    else:
        new_width = 256
        new_height = int(height / width * 256)
    image = image.resize((new_width,new_height))

    # center crop
    width, height = image.size
    new_width = 224
    new_height = 224
    image = image.crop(((width - new_width)/2,
                        (height - new_height)/2,
                        (width + new_width)/2,
                        (height + new_height)/2))

    # normalize
    np_image = np.array(image)
    np_image = np_image/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # transpose
    np_image = np_image.transpose((2,0,1))

    # to tensor
    tensor_image = torch.from_numpy(np_image)
    tensor_image = torch.reshape(tensor_image, (1, 3, 224, 224))
    tensor_image = tensor_image.to(torch.float32)
    return tensor_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
#     print(image.shape)
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    if title is not None:
        ax.set_title(title)
    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    with Image.open(image_path) as image:
        image = process_image(image)

        model.eval()
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, indices = ps.topk(topk, dim=1)
        probs = [p.item() for p in probs.view(-1)]
        classes = [str(i.item()+1) for i in indices.view(-1)]
    return image[0], probs, classes

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image with probability of that name'
    )
    parser.add_argument('image_path', help='an image to predict')
    parser.add_argument('checkpoint', help='model checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='return top K most likely classes')
    parser.add_argument('--category_name', default='cat_to_name.json',
                        help='Use a mapping of categories to real names')
    args = parser.parse_args()
    print('parse arguments:\n\t', args, '\n')

    # model
    checkpoint = args.checkpoint
    category_name = args.category_name
    model, criterion, optimizer = load_checkpoint(checkpoint)
    cat_to_name = load_categories(category_name)

    # predict
    image_path = args.image_path
    top_k = args.top_k
    image, probs, classes = predict(image_path, model, top_k)
    flowers = [cat_to_name[c] for c in classes]
    print('predict:\n\t', probs, '\n\t', flowers)

    # visulize
    fig, (ax1, ax2) = plt.subplots(figsize=(6,6), nrows=2)
    imshow(image, ax1, flowers[0])
    ax2.barh(flowers, probs)
    plt.show()
