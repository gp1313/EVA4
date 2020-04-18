from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_transforms():
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomVerticalFlip(),
                                           transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std),
                                                        transforms.RandomErasing()])
    test_transforms = transforms.Compose([transforms.ToTensor(), 
                                                        transforms.Normalize(mean, std)])

    return train_transforms, test_transforms
    
def get_dataloder(root_path='./data', args={}):
    
    train_transforms, test_transforms = get_transforms()

    train_dataset = ImageFolder(root_path + '/train', transform=train_transforms)
    test_dataset = ImageFolder(root_path + '/test', transform=test_transforms)

    #args = {'pin_memory' : True, 'num_workers': 1}
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **args)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, **args)

    return train_loader, test_loader