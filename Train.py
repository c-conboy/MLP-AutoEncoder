import datetime
import torch


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training ...')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch', epoch)
        loss_train = 0.0

        for imgs, labels in train_loader: #for each batch of 2048 images
            imgs2 = torch.zeros(imgs.size()[0], 784) #Create an empty tesnor that 2048 by 28*28
            for i in range(imgs.size()[0]): #for each image in batch
                imgs2[i] = imgs[i].flatten()#Flatten image at index and then put into
            #Flattened
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