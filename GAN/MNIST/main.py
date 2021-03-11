import torch 
from torch import optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from model import Generator,Discriminator
import numpy as np
from matplotlib import cm
from torch import nn 
from random import randint 
import os 
batches=128
input_features=128
generator=Generator(input_features)
discriminator=Discriminator()
is_cuda=torch.cuda.is_available()
if is_cuda:
    generator=generator.cuda()
    discriminator=discriminator.cuda()

criterion=nn.BCELoss()
g_optimizer=optim.Adam(generator.parameters(),lr=0.0002)
d_optimizer=optim.Adam(discriminator.parameters(),lr=0.0002)

def load_MNIST_dataset():
    folder='./dataset/'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)),
        torch.flatten
    ])
    data= datasets.MNIST(
            root=folder,
            train=True,
            transform=transform,
            download=True
        )

    return DataLoader(dataset=data,batch_size=batches,shuffle=True)

def generate_noise(batch_size,in_features):
    if is_cuda:
        return torch.rand(batch_size,in_features).cuda()
    return torch.randn(batch_size,in_features)

def generate_ones(batch_size):
    if is_cuda:
        return torch.ones(batch_size,1).cuda()
    return torch.ones(batch_size,1)

def generate_zeros(batch_size):
    if is_cuda:
        return torch.zeros(batch_size,1).cuda()
    return torch.zeros(batch_size,1)

def train_discriminator(x,batch_size):
    d_optimizer.zero_grad()
    predict_real=discriminator(x)
    error_real=criterion(predict_real,generate_ones(batch_size))
    error_real.backward()

    fake_samples=generator(generate_noise(batch_size,in_features=128))
    predict_fake=discriminator(fake_samples)
    error_fake=criterion(predict_fake,generate_zeros(batch_size))
    error_fake.backward()

    d_optimizer.step()
    return error_fake+error_real,predict_fake,predict_real


def train_generator(batch_size):
    g_optimizer.zero_grad()
    generate_sample=generator(generate_noise(batch_size,in_features=128))
    predict=discriminator(generate_sample)
    error=criterion(predict,generate_ones(batch_size))
    error.backward()
    g_optimizer.step()
    return error

def plot_image():
    if is_cuda:
        img=generator(generate_noise(1,128)).cpu().detach().view(28,28).numpy(),
        plt.imshow(img,cmap=cm.gray)

    else:
        img=generator(generate_noise(1,128)).detach().view(28,28).numpy()
        plt.imshow(img,cmap=cm.gray)

    path=os.path.join(os.getcwd(),'Result')
    if not os.path.exist(path):
        os.mkdir(path)
    plt.imsave('Result/result{}.png'.format(randint(1,1000)),img)

def show_result():
    n_rows=3
    n_cols=3
    plt.figure(figsize=(10,10))
    for i in range(n_rows*n_cols):
        plt.subplot(n_rows,n_cols,i+1)
        plot_image()
    plt.savefig('result.png')
    plt.show()

def plot_loss(dis_error_his,gen_error_his,epochs):
    plt.plot(range(epochs),dis_error_his,label='discriminator')
    plt.plot(range(epochs),gen_error_his,label='generator')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.savefig('loss.png')
    plt.show()

def train():
    data_loader=load_MNIST_dataset()
    num_batcher=len(data_loader)
    epochs=2000
    dis_error_his=[]
    gen_error_his=[]
    for epoch in range(epochs):
        g_errs=0
        d_errs=0
        dx=0
        gx=0

        for batch_sample in data_loader:
            x_sample=batch_sample[0].cuda() if is_cuda else batch_sample[0]
            batch_size=x_sample.size(0)
            dis_error,predict_fake,predict_real=train_discriminator(x_sample,batch_size)
            g_error=train_generator(batch_size)
            g_errs+=g_error
            d_errs+=dis_error
            dx+=predict_real.mean()
            gx+=predict_fake.mean()

    print("Epoch:{}---Dx:{}---Gx:{}---Derr:{}---Gerr:{}".format(epoch,dx/num_batcher,gx/num_batcher,d_errs/num_batcher,g_errs/num_batcher))
    dis_error_his.append(dx/num_batcher)
    gen_error_his.append(gx/num_batcher)
    if epoch%100==0:
        plot_image()
    
    if epoch%500==0:
        torch.save(generator,'generator{}.pth'.format(epoch))
        torch.save(discriminator,'discriminator{}.pth'.format(epoch))

    plot_loss(dis_error_his,gen_error_his,epochs)
    show_result()


if __name__=="__main__":
    train()

