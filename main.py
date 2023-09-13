import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import Model
import Train
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.optim.lr_scheduler as lr_scheduler

train_transform = transforms.Compose([transforms.ToTensor()])

'''
idx = int(input("Enter your value: "))
plt.imshow(train_set.data[idx], cmap='gray')
plt.show()
'''
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

rate_learning = 1e-3
model = Model.autoencoderMLP4Layer()
optim = optim.Adam(model.parameters(), lr=rate_learning)
loss = nn.MSELoss()
train_dataloader = DataLoader(train_set, batch_size=2048, shuffle=True)
scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.3, total_iters=10)
device = torch.device('cpu')
Train.train(n_epochs=50, optimizer=optim, model=model, loss_fn=loss, train_loader=train_dataloader, scheduler=scheduler, device=device)