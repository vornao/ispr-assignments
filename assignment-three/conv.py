import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import tqdm auto progress bar
from tqdm.auto import tqdm


VAL_STRING_TEMPLATE = \
    "Epoch {epoch:4d} \
    | Training Loss: {train_loss:3f} \
    | Training Accuracy: {train_acc:3f} \
    | Validation Loss: {val_loss:3f} \
    | Validation Accuracy: {val_acc:3f}"


TR_STRING_TEMPLATE = \
    "Epoch {epoch:4d}\
    | Training Loss: {train_loss:3f}\
    | Training Accuracy: {train_acc:3f}"


def train(model, device, train_loader, optimizer, criterion, epoch, *args, **kwargs):
    """
    Train the model
    :param: model: the model to train
    :param: device: the device to use
    :param: train_loader: the train loader
    :param: optimizer: the optimizer
    :param: criterion: the loss function
    :param: epoch: the current epoch
    :param: args: additional arguments
    :param: kwargs: additional keyword arguments
    :param: val_loader: the validation loader

    :return: the loss and accuracy
    """

    model.to(device)
    model.train()
    epoch_loss = 0
    correct = 0

    # take validation set from kwargs
    val_loader = kwargs.get("val_loader", None)


    for d in train_loader:

        input, label = d
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    train_loss = epoch_loss/len(train_loader)
    train_acc = correct/len(train_loader.dataset)

    if epoch % 1 == 0:
        if val_loader is not None :
                val_loss, val_acc = test(model, device, val_loader, criterion, epoch)
                print(VAL_STRING_TEMPLATE.format(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))
        else:
            print(TR_STRING_TEMPLATE.format(epoch=epoch, train_loss=train_loss, train_acc=train_acc))
  
    if val_loader is not None:
        val_loss, val_acc = test(model, device, val_loader, criterion, epoch)
        return {'epoch':epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
    else:
        return {'epoch':epoch, "train_loss": train_loss, "train_acc": train_acc}

def test(model, device, test_loader, criterion, epoch, verbose=False):
    """
    Test the model
    :param: model: the model to test
    :param: device: the device to use
    :param: test_loader: the test loader
    :param: criterion: the loss function
    :param: epoch: the current epoch

    :return: the loss and accuracy
    """
    
    model.eval()
    epoch_loss = 0
    correct = 0
    with torch.no_grad():
        for i, d in enumerate(test_loader):

            input, label = d
            input, label = input.to(device), label.to(device)

            output = model(input)
            loss = criterion(output, label)
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    if verbose:
        print(f"Epoch {epoch} | Test Loss: {epoch_loss/len(test_loader)} | Test Accuracy: {correct/len(test_loader.dataset)}")

    return epoch_loss/len(test_loader), correct/len(test_loader.dataset)

class BlockCNN(nn.Module):
    # define sequential model
    def __init__(self):
        super(BlockCNN, self).__init__()

        # first convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)              # padding to keep size
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)             
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)            
        self.conv4 = nn.Conv2d(128, 256, 3)                      # no padding
        # MLP classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # first block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(self.pool(x))

        # second block
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(self.pool(x))

        # third block
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(self.pool(x))

        # fourth block
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(self.pool(x))
        
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x