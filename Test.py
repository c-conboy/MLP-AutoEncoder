import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from Model import autoencoderMLP4Layer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)

test_transform = transforms.Compose([transforms.ToTensor()])
test_set = MNIST('./data/mnist', train=True, download=True, transform=test_transform)


model = autoencoderMLP4Layer().to(device)
model.load_state_dict(torch.load("./MLP.8.pth"))
model.eval()


def test_with_noise():
    idx = int(input("Enter your value: "))
    with torch.no_grad():
        test_set_flattened = torch.zeros(test_set.data.size()[0], 784, dtype=torch.float32)
        i = 0

        for img in test_set.data:
            test_set_flattened[i] = torch.flatten(img)
            i = i + 1

        x = test_set_flattened[idx]
        mean, std, var = torch.mean(x), torch.std(x), torch.var(x)
        x = (x - mean) / std
        noise = torch.rand(x.size(), dtype=torch.float32)
        xn = x + noise
        xn = xn.to(device)
        pred = model(xn)


    f = plt.figure()
    f.add_subplot(1, 3, 1)
    x = x.unflatten(-1,(28,28))
    plt.imshow(x, cmap='gray')
    f.add_subplot(1, 3, 2)
    xn = xn.unflatten(-1,(28,28))
    plt.imshow(xn, cmap='gray')
    f.add_subplot(1, 3, 3)
    pred = pred.unflatten(-1,(28,28))
    plt.imshow(pred, cmap='gray')
    plt.show()

def get_image():
    idx = int(input("Enter your value: "))
    return test_set.data[idx]

def flatten(x):
    test_set_flattened = torch.zeros(1, 784, dtype=torch.float32)
    test_set_flattened[0] = torch.flatten(x)
    return test_set_flattened[0]

def normalise(x):
    mean, std, var = torch.mean(x), torch.std(x), torch.var(x)
    x = (x - mean) / std
    return x

def test_interpolate():
    with torch.no_grad():
        f = plt.figure()
        n_steps = 10

        x = get_image()
        f.add_subplot(2, n_steps, 1)
        plt.imshow(x, cmap='gray')
        x = flatten(x)
        x = normalise(x)
        x = x.to(device)
        f.add_subplot(2, n_steps, 2)
        plt.imshow(model(x).unflatten(-1,(28,28)), cmap='gray')

        y = get_image()
        f.add_subplot(2, n_steps, 3)
        plt.imshow(y, cmap='gray')
        y = flatten(y)
        y = normalise(y)
        y = y.to(device)
        f.add_subplot(2, n_steps, 4)
        plt.imshow(model(y).unflatten(-1,(28,28)), cmap='gray')

        bx = model.encode(x)
        by = model.encode(y)

        interpolations = []
        weights = torch.linspace(1,0,n_steps)

        for weight in weights:
            interpolationAtWeight = bx*weight + by*(1-weight)
            interpolations.append(interpolationAtWeight)

        decodedInterpolations = []
        for interpolation in interpolations:
            decodedInterpolations.append(model.decode(interpolation))

        for i in range(n_steps):
            f.add_subplot(2, n_steps, i+11)
            decodedInterpolations[i] = decodedInterpolations[i].unflatten(-1,(28,28))
            plt.imshow(decodedInterpolations[i], cmap='gray')
        plt.show()

test_interpolate()


