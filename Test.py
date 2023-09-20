import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from Model import autoencoderMLP4Layer

#Choose a device to do computation
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

#Run model on chosen device
model = autoencoderMLP4Layer().to(device)
model.load_state_dict(torch.load("./MLP.8.pth")) #what does this line do?
model.eval()

def single_test():
    with torch.no_grad():
        x = get_image()
        x = flatten(x)
        x = normalise(x)
        x = x.to(device)
        pred = model(x)

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    x = x.unflatten(-1,(28,28))
    plt.imshow(x, cmap='gray')
    f.add_subplot(1, 2, 2)
    pred = pred.unflatten(-1,(28,28))
    plt.imshow(pred, cmap='gray')
    plt.show()


def test_with_noise():
    with torch.no_grad():
        x = get_image()
        x = flatten(x)
        x = normalise(x)
        noise = torch.rand(x.size(), dtype=torch.float32)
        xn = x + noise
        xn = xn.to(device)
        pred = model(xn)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    x = x.unflatten(-1,(28,28))
    plt.imshow(x, cmap='gray') #plot original image
    f.add_subplot(1, 3, 2)
    xn = xn.unflatten(-1,(28,28)) #plot image with noise
    plt.imshow(xn, cmap='gray')
    f.add_subplot(1, 3, 3)
    pred = pred.unflatten(-1,(28,28)) #plot image denoised
    plt.imshow(pred, cmap='gray')
    plt.show()

def get_image():
    idx = int(input("Enter your value: "))
    return test_set.data[idx]

#Reshape image into a one-dimensional tensor
def flatten(x):
    test_set_flattened = torch.zeros(1, 784, dtype=torch.float32)
    test_set_flattened[0] = torch.flatten(x)
    return test_set_flattened[0]

#why is stuff commented out here?
def normalise(x):
    #mean, std, var = torch.mean(x), torch.std(x), torch.var(x)
    #x = (x - mean) / std
    x = x / 255
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

        bx = model.encode(x) #bottleneck output of image 1 (x)
        by = model.encode(y) #bottleneck output of image 2 (y)

        interpolations = []
        weights = torch.linspace(1,0,n_steps)

        #linearly interpolate (e.g. 0.1x and 0.9y, then 0.2x, 0.8y, and so on)
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

print("Input an image index for a single test")
single_test()
print("Input an image index for a single test with noise")
test_with_noise()
print("Input two image indices for a interpolation test")
test_interpolate()


