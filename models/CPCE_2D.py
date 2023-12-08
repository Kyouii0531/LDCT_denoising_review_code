import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision.models import vgg19
import numpy as np

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=0)

        self.conv1x1 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoder
        reisdual_1 = x

        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))

        # out2_copy = out2
        out3 = self.relu(self.conv3(out2))
        # out3_copy = out3
        out4 = self.relu(self.conv4(out3))

        # decoder
        out5 = self.tconv1(out4)

        out5 = self.relu(torch.cat((out5, out3), dim=-3))
        out5 = self.relu(self.conv1x1(out5))

        out6 = self.tconv2(out5)
        out6 = self.relu(torch.cat((out6, out2), dim=-3))
        out6 = self.relu(self.conv1x1(out6))

        out7 = self.tconv3(out6)
        out7 = self.relu(torch.cat((out7, out1), dim=-3))
        out7 = self.relu(self.conv1x1(out7))

        out8 = self.relu(self.tconv4(out7) + reisdual_1)

        return out8

class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256 * self.output_size * self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class CPCE_2D(nn.Module):

    def __init__(self, input_size=64):
        super(CPCE_2D, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator(input_size)
        self.feature_extractor = VGG_FeatureExtractor()
        self.p_criterion = nn.MSELoss()

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + 10 * gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def mse_loss(self, x, y):
        fake = self.generator(x)
        mse_loss = nn.functional.mse_loss(fake, y)

        return mse_loss








