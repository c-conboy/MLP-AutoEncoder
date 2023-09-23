import datetime
import torch
from matplotlib import pyplot as plt
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import Model
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from torchsummary import summary


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training ...')
    model.train()
    losses_train = []
    losses_data = torch.zeros(50)
    for epoch in range(1, n_epochs+1):
        print('epoch', epoch)
        loss_train = 0.0

        for imgs, labels in train_loader: #for each batch of 2048 images
            imgs2 = torch.zeros(imgs.size()[0], 784) #Create an empty tesnor that's 2048*28*28
            for i in range(imgs.size()[0]): #for each image in the batch
                imgs2[i] = imgs[i].flatten()#Flatten image at the index
            imgs2 = imgs2.to(device=device)
            outputs = model(imgs2)
            loss = loss_fn(outputs, imgs2) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
        losses_data[epoch-1] = loss_train/len(train_loader)

    return losses_data


parser = argparse.ArgumentParser()
parser.add_argument("-z", "--BottleneckSize")
parser.add_argument("-e", "--Epochs")
parser.add_argument("-b", "--BatchSize")
parser.add_argument("-s", "--Checkpoint")
parser.add_argument("-p", "--LossChart")
args = parser.parse_args()

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

rate_learning = 1e-3
model = Model.autoencoderMLP4Layer(N_bottleneck=int(args.BottleneckSize))
optim = optim.Adam(model.parameters(), lr=rate_learning)
loss = nn.MSELoss()
train_dataloader = DataLoader(train_set, batch_size=int(args.BatchSize), shuffle=True)
scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.3, total_iters=10)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)
loss_data = train(n_epochs=50, optimizer=optim, model=model, loss_fn=loss, train_loader=train_dataloader, scheduler=scheduler, device=device)
torch.save(model.state_dict(), args.Checkpoint)
plt.plot(loss_data)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(args.LossChart)

