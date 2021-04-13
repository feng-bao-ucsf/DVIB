# from training.multiview_infomax import MVInfoMaxTrainer
from utils.schedulers import ExponentialScheduler
from utils.schedulers import LinearScheduler
from training.base import MergeTrainer
from utils.modules import MIEstimator, Encoder, Decoder, Feature_extractor, Alex_extractor, Alex_extractor_avg, Alex_extractor_fea
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torchvision import models
###############
# DVIB Trainer #
###############


class DVIBTrainer(MergeTrainer):
    def __init__(self, beta=1,
                 lambda_start_value=1e-5, lambda_end_value=1,
                 lambda_n_iterations=100000, lambda_start_iteration=50000,
                 miest_lr=1e-4,**params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(DVIBTrainer, self).__init__(**params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta = beta
        self.lambda_scheduler = ExponentialScheduler(start_value=lambda_start_value, end_value=lambda_end_value,
                                                   n_iterations=lambda_n_iterations, start_iteration=lambda_start_iteration)
        #self.lambda_scheduler = LinearScheduler(start_value=lambda_start_value, end_value=lambda_end_value,
        #                                             n_iterations=lambda_n_iterations, start_iteration=lambda_start_iteration)

        # Initialization of the mutual information estimation network
        self.mi_estimator_x = MIEstimator(self.z_dim, self.z_dim)
        self.mi_estimator_y = MIEstimator(self.z_dim, self.z_dim)

        # Adding the parameters of the estimator to the optimizer

        self.opt.add_param_group(
            {'params': self.mi_estimator_x.parameters(), 'lr': miest_lr},
            )
        self.opt.add_param_group(
            {'params': self.mi_estimator_y.parameters(), 'lr': miest_lr},
        )

        # Defining the prior distribution as a factorized normal distribution
        self.mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)
        self.prior = Normal(loc=self.mu, scale=self.sigma)
        self.prior = Independent(self.prior, 1)


        # if x and y follow the same distribution, encoders for shared representation can share parameters
        # self.encoder_y_s = self.encoder_x_s # reuse
        self.mu2 = nn.Parameter(torch.full((self.z_dim,),1.), requires_grad=False)
        self.mu3 = nn.Parameter(torch.full((self.z_dim,),2.), requires_grad=False)
        self.prior2 = Normal(loc=self.mu2, scale=self.sigma)
        self.prior2 = Independent(self.prior2, 1)
        self.prior3 = Normal(loc=self.mu3, scale=self.sigma)
        self.prior3 = Independent(self.prior3, 1)

        ####
        #create pretrained resnet18, Alexnet
        ####

        self.res = Feature_extractor(output_layer='avgpool')
        self.alex = Alex_extractor()



        ###start lambda
        self.labda_start = 0
    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        x, y, _, _ = data
        x = self.res(x)
        y = self.res(y)
        #x = self.alex(x)
        #y = self.alex(y)

        # Read new dataset, views v1 and v2
        # Encode a batch of data
        p_z_xs_given_x = self.encoder_x_s(x)  # [z_dim * 2]
        p_z_xp_given_x = self.encoder_x_p(x)  # [z_dim * 2]
        p_z_ys_given_y = self.encoder_y_s(y)  # [z_dim * 2]
        p_z_yp_given_y = self.encoder_y_p(y)  # [z_dim * 2]

        # Sample from the posteriors with reparametrization
        z_xs = p_z_xs_given_x.rsample()
        z_xp = p_z_xp_given_x.rsample()
        z_ys = p_z_ys_given_y.rsample()
        z_yp = p_z_yp_given_y.rsample()

        # Reconstruction loss from private and shared latents
        q_x_given_z_xs = self.decoder_x_s(z_xs)
        q_x_given_z_xp = self.decoder_x_p(z_xp)
        q_y_given_z_ys = self.decoder_y_s(z_ys)
        q_y_given_z_yp = self.decoder_y_p(z_yp)

        reconstruct_loss_xs = -q_x_given_z_xs.log_prob(x.view(x.shape[0], -1))
        reconstruct_loss_xp = -q_x_given_z_xp.log_prob(x.view(x.shape[0], -1))
        reconstruct_loss_ys = -q_y_given_z_ys.log_prob(y.view(y.shape[0], -1))
        reconstruct_loss_yp = -q_y_given_z_yp.log_prob(y.view(y.shape[0], -1))

        neg_I_x = reconstruct_loss_xs.mean() + reconstruct_loss_xp.mean()
        neg_I_y = reconstruct_loss_ys.mean() + reconstruct_loss_yp.mean()

        # Mutual information estimation between private and shared 
        # representations from the same view
   

        mi_gradient_x, mi_estimation_x = self.mi_estimator_x(z_xs, z_ys)
        mi_gradient_x = mi_gradient_x.mean()
        mi_estimation_x = mi_estimation_x.mean()

        mi_gradient_y, mi_estimation_y = self.mi_estimator_y(z_ys, z_xs)

        mi_gradient_y = mi_gradient_y.mean()
        mi_estimation_y = mi_estimation_y.mean()

        mi_gradient = mi_gradient_x + mi_gradient_y
        mi_estimation = mi_estimation_x + mi_estimation_y

        # Upper bound of mutual information between different views
        pos_I_y_zxp = p_z_yp_given_y.log_prob(z_yp) - self.prior2.log_prob(z_xp)
        pos_I_x_zyp = p_z_xp_given_x.log_prob(z_xp) - self.prior3.log_prob(z_yp)

        pos_beta_I = pos_I_y_zxp.mean() + pos_I_x_zyp.mean()

        # Update the value of beta according to the policy

        labda = self.lambda_scheduler(self.iterations - self.labda_start)


        # Logging the components
        self._add_loss_item('loss/neg_I_x', neg_I_x.item())
        self._add_loss_item('loss/neg_I_y', neg_I_y.item())
        self._add_loss_item('loss/I_z1_z2', mi_estimation.item())
        self._add_loss_item('loss/softplus_I_12', mi_gradient.item())
        #self._add_loss_item('loss/ckl', neg_ckl.item())
        self._add_loss_item('loss/I_beta', pos_beta_I.item())
        self._add_loss_item('loss/beta', self.beta)
        self._add_loss_item('loss/lambda', labda)

        # Computing the loss function
        loss = neg_I_x + neg_I_y - labda*mi_gradient + beta*pos_beta_I

        return loss
