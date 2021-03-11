import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,input_feature):
        super(Generator,self).__init__()
        self.input_feature=input_feature
        self.out_feature=28*28

        self.hidden0=nn.Sequential(
            nn.Linear(self.input_feature,256),
            nn.LeakyReLU(0.2)
        )

        self.hidden1=nn.Sequential(
                nn.Linear(256,512),
                nn.LeakyReLU(0.2)
        )

        self.hidden2=nn.Sequential(
                nn.Linear(512,1024),
                nn.LeakyReLU(0.2)
        )

        self.out=nn.Sequential(
                nn.Linear(1024,self.out_feature),
                nn.Tanh()
        )

    def forward(self,x):
        x=self.hidden0(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.n_features=28*28
        self.out_features=1

        self.hidden0=nn.Sequential(
            nn.Linear(self.n_features,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1=nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2=nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out=nn.Sequential(
            nn.Linear(256,self.out_features),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.hidden0(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        return self.out(x)

