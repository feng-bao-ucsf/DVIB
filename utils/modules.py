import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torchvision import models

# Pretrained Feature Extractor, resnet18
class Feature_extractor (nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x
#Feature_Extractor, Alexnet, output_layer = -3 and -6 are dropout layers
class Alex_extractor(nn.Module):
    def __init__(self, output_layer = None):
        super(Alex_extractor, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(self.pretrained.features.children())
        )
        self.avgpool = nn.Sequential(
            *list(self.pretrained.avgpool.modules())
        )
        self.classifier = nn.Sequential(
            *list(self.pretrained.classifier.children())[:output_layer]
        )
        self.pretrained = None
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class Alex_extractor_avg(nn.Module):
    def __init__(self, output_layer = None):
        super(Alex_extractor_avg, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(self.pretrained.features.children())
        )
        self.avgpool = nn.Sequential(
            *list(self.pretrained.avgpool.modules())
        )
        self.pretrained = None
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Alex_extractor_fea(nn.Module):
    def __init__(self, output_layer = None):
        super(Alex_extractor_fea, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(self.pretrained.features.children())[:output_layer]
        )
        self.pretrained = None
    def forward(self, x):
        x = self.features(x)
        return x
# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            #nn.Linear(28 * 28, 1024),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        #CNN output is batchsize * 512 * 1 * 1 , x.view(xize(0), -1), converse to batchsize * 1 * 512
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            #nn.Linear(1024, 28 * 28)
            nn.Linear(1024, 512)
        )

    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)

# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1