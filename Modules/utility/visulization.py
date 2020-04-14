from matplotlib import pyplot as plt
import numpy as np
import torch

def display_graphs(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs[0, 0].plot(train_acc)
    axs[0, 0].set_title('train_acc')
    axs[0, 1].plot(train_losses)
    axs[0, 1].set_title('train_loss')
    axs[1, 0].plot(test_acc)
    axs[1, 0].set_title('test_acc')
    axs[1, 1].plot(test_losses)
    axs[1, 1].set_title('test_loss')
    plt.show()

def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        tensor = tensor.view(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean).squeeze()

def display_images(images, labels, idx_to_class):
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    fig, axs = plt.subplots(4, 8, figsize=(15, 10))
    for i in range(32):
        r = int(i / 8)
        c = int(i % 8)
        label = labels[i]
        img = images[i]
        
        img = denormalize(img, mean, std)

        np_img = img.numpy().transpose(1, 2, 0)
        axs[r, c].imshow(np_img)
        axs[r, c].set_title(idx_to_class[label.item()])
    plt.show()