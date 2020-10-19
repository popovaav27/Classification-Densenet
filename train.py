import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import utils
import torch.nn.functional as F
import argparse

#ArgParse
parser = argparse.ArgumentParser(description='Classification Densenet-201')
parser.add_argument('--data_dir_train', type=str, required = True, help='Train dir - with subdirs images ans masks')
parser.add_argument('--data_dir_test', type=str, required = True, help='Test dir - with subdirs images ans masks')
parser.add_argument('--image_size', type=int,default=224, help='Image size - for cropping the images to nxn images')
parser.add_argument('--n_classes', type=int,default=12, help='Number of classes - number of classes')
parser.add_argument('--batch_size', type=int,default=16, help='Batch size')
parser.add_argument('--epochs', type=int,default=1000, help='Number of Epochs')
args = parser.parse_args()


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = args.epochs
num_classes = args.n_classes
batch_size = args.batch_size
learning_rate = 0.001
image_size = (args.image_size, args.image_size)

tr1 = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


tr2 = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

train_dataset = torchvision.datasets.ImageFolder(args.data_dir_train, transform=tr1)
test_dataset = torchvision.datasets.ImageFolder(args.data_dir_test, transform=tr2)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


model = torchvision.models.densenet201(True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(1920, num_classes)

model.to(device)

criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)


trainedModel, accNetCurrent = utils.train(model, criterion, optimizer, train_loader, test_loader, num_epochs, device, num_classes, True)

print('acc: {}'.format(accNetCurrent))


# Save the model
torch.save(trainedModel.state_dict(), 'result.ckpt')

