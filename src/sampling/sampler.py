import torch
import numpy as np
from .likelihood import *
from .prior import *
from .posterior import *
from src.utils.helpers import *
from src.utils.logging_setup import setup_logging

class Sampling:
    """
    A class for handling sampling operations in a generative model.

    Args:
        G (nn.Module): Generator model.
        GMM (nn.Module): GMM prior model.
        sampler (callable): Function for sampling latent variables.
        likelihood (str): Likelihood function type ('gaussian' or 'poisson').
        latent_dim (int): Dimensionality of the latent space.
        pushforward (callable): Function to apply after generating samples.
        time (torch.Tensor): Time points (used in generating samples).

    Attributes:
        device (torch.device): Device for computation (cuda or cpu).
        log_likelihood_sigma (torch.Tensor): Standard deviation for the Gaussian likelihood.
        sampler (callable): Function for sampling latent variables.
        latent_dim (int): Dimensionality of the latent space.
        dataset (str): Dataset type.
        G (nn.Module): Initialized Generator model.
        GMM (nn.Module): Initialized GMM model.
        average (int): Number of steps for averaging.
        lrGMM (float): Learning rate for GMM.
        lrG (float): Learning rate for Generator.
        lrGMM_decay (float): Decay rate for GMM learning rate.
        lrG_decay (float): Decay rate for Generator learning rate.
        true_coeffs (torch.Tensor): True coefficients (if applicable).
        time (torch.Tensor): Time points (used in generating samples).
        testing (bool): Flag for testing mode.
    """

    def __init__(self, G, GMM, sampler, likelihood, pushforward, time):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_likelihood_sigma = torch.tensor(0.3, device=self.device)
        self.sampler = sampler
        self.likelihood = likelihood
        self.dataset = 'poisson'
        self.plot = False
        self.pushforward = pushforward
        self.G = self.init_model(G)
        self.GMM = GMM.to(self.device)
        self.average = 30
        self.lrGMM = 1e-3
        self.lrG = 1e-4
        self.lrGMM_decay = 0.998
        self.lrG_decay = 0.998
        self.true_coeffs = None
        self.time = time
        self.testing = False

    def init_model(self, model):
        """
        Initialize the given model with Xavier weights and move it to the specified device.

        Args:
            model (nn.Module): Neural network model.

        Returns:
            nn.Module: Initialized model.
        """
        return model.apply(weights_init_xavier).to(self.device)

    def loss(self, **kwargs):
        """
        Compute the loss components for training the generative model.

        Args:
            z_post (torch.Tensor): Latent variables sampled from the posterior distribution.
            x (torch.Tensor): True data.
            means (torch.Tensor): Means of the Gaussian Mixture Model (GMM) components.
            lower_cholesky (torch.Tensor): Lower Cholesky decomposition of the GMM covariance matrices.
            weights (torch.Tensor): Weights of the GMM components.

        Returns:
            torch.Tensor: Negative log prior loss.
            torch.Tensor: Negative log likelihood loss.
            torch.Tensor: Posterior latent variables.
        """
        z_post = kwargs['z_post']
        x = kwargs['x']
        means = kwargs['means']
        lower_cholesky = kwargs['lower_cholesky']
        weights = kwargs['weights']
        time = kwargs['time']

        loss_gmm = -log_prior(z=z_post.detach(), 
                              means=means, 
                              lower_cholesky=lower_cholesky, 
                              weights=weights).mean().to(self.device)
        loss_g, smoothness = log_likelohood(x=x,
                                 z=z_post.detach(),
                                 time=time,
                                 model=self.G,
                                 pushforward=self.pushforward,
                                 log_likelihood_sigma=self.log_likelihood_sigma,
                                 testing=self.testing,
                                 num_samples=1000)
        
        loss_g = -loss_g.mean().to(self.device) #+ smoothness #+ torch.mean(self.G.log_sigma(z_post.detach())**2)
        
        return loss_gmm, loss_g

    def train(self, **kwargs):
        """
        Train the generative and encoder models using the specified training parameters.

        Args:
            data (torch.Tensor): Training dataset.
            n_iter (int): Number of training epochs.
            batch_size (int): Batch size for training.
            num_steps_post (int): Number of Langevin steps for sampling from the posterior.
            step_size_post (float): Step size for Langevin dynamics in the posterior.

        Returns:
            torch.Tensor: Updated posterior latent variables.
        """
        data = kwargs['data']
        n_iter = kwargs['n_iter']
        batch_size = kwargs['batch_size']
        num_steps_post = kwargs['num_steps_post']
        step_size_post = kwargs['step_size_post']

        self.optG = torch.optim.Adam(self.G.parameters(), lr=self.lrG, weight_decay=1e-5, betas=(0.5, 0.999))
        self.optGMM = torch.optim.Adam(self.GMM.parameters(), lr=self.lrGMM, weight_decay=1e-5, betas=(0.5, 0.999))
        self.lr_scheduleGMM = torch.optim.lr_scheduler.ExponentialLR(self.optGMM, self.lrGMM_decay)
        self.lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(self.optG, self.lrG_decay)

        dir_name = f'{self.dataset}/{self.sampler}_{self.likelihood}/{num_steps_post}_{step_size_post}_lrGMM_{self.lrGMM}_lrG_{self.lrG}'
        makedir(dir_name)

        logger = setup_logging('job0', dir_name, console=True)
        parameters = ['GMM', num_steps_post, step_size_post]
        logger.info(f'Training for {n_iter} epochs with sampler {self.sampler}, {self.likelihood} likelihood and [num_steps_prior, step_size_prior, num_steps_post, sampler]: {parameters}')
        logger.info(f'Details: lr_scheduleGMM {get_lr(self.optGMM)}, lr_scheduleG {get_lr(self.optG)}, GMM.components {self.GMM.components}, GMM.dimensions {self.GMM.dimensions}')
        logger.info(self.GMM)
        logger.info(self.G)

        total_param_e = sum(p.numel() for p in self.GMM.parameters())
        trainable_params_e = sum(p.numel() for p in self.GMM.parameters() if p.requires_grad)
        logger.info(f'Trainable parameters for GMM: {trainable_params_e}/{total_param_e}')

        total_param_g = sum(p.numel() for p in self.G.parameters())
        trainable_params_g = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        logger.info(f'Trainable parameters for Generator: {trainable_params_g}/{total_param_g}')

        self.loss_gmm_save = []
        self.loss_g_save = []

        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(1, n_iter + 1):
            logger.info(f'Epoch {epoch}/{n_iter}')
            for batch, x in enumerate(dataloader, 0):
                torch.autograd.set_detect_anomaly(True)
                logger.info(f'Batch: {batch+1}/{len(dataloader)}')

                x = x.view(batch_size, -1).to(self.device)
                prior_final, means, lower_cholesky, weights = self.GMM(batch_size)
                post_final = langevin(x=x, z=prior_final, means=means, lower_cholesky=lower_cholesky, weights=weights, time=self.time, step_size=step_size_post, num_steps=num_steps_post, model=self.G, pushforward=self.pushforward, log_likelihood_sigma=self.log_likelihood_sigma, plot=self.plot)[0]

                self.optG.zero_grad()
                self.optGMM.zero_grad()

                loss_gmm, loss_g = self.loss(z_post=post_final, x=x, means=means, lower_cholesky=lower_cholesky, weights=weights, time=self.time)
                loss_g.backward()
                self.optG.step()

                loss_gmm.backward()
                self.optGMM.step()

                if torch.isnan(loss_gmm) or torch.isnan(loss_g):
                    raise ValueError('Invalid GMM loss.')

                self.loss_gmm_save.append(loss_gmm.detach().cpu().data.numpy())
                self.loss_g_save.append(loss_g.detach().cpu().data.numpy())

                np.save(f'{dir_name}/chains/loss_gmm.npy', np.array(self.loss_gmm_save))
                np.save(f'{dir_name}/chains/loss_g.npy', np.array(self.loss_g_save))

                logger.info(f'Loss GMM: {loss_gmm.item():.3f}')
                logger.info(f'Loss G: {loss_g.item():.3f}')

            if epoch % 1 == 0:
                save_model(dir_name, epoch, 'last_model', self.GMM, self.optGMM, self.lr_scheduleGMM, self.G, self.optG, self.lr_scheduleG)
                logger.info(f'Saved model at epoch {epoch}')
            
            if epoch % 100 == 0:
                save_model(dir_name, epoch, f'{epoch}_model', self.GMM, self.optGMM, self.lr_scheduleGMM, self.G, self.optG, self.lr_scheduleG)
                logger.info(f'Saved model at epoch {epoch}')
            self.lr_scheduleGMM.step()
            self.lr_scheduleG.step()

        np.save(f'{dir_name}/chains/loss_gmm.npy', np.array(self.loss_gmm_save))
        np.save(f'{dir_name}/chains/loss_g.npy', np.array(self.loss_g_save))

    def generate_samples(self, **kwargs):
        """
        Generate samples using Langevin dynamics and the trained generative model.

        Args:
            data (torch.Tensor): True data.
            num_gen_samples (int): Number of samples to generate.
            n_iter (int): Number of training iterations.
            num_steps_post (int): Number of Langevin steps for the posterior distribution.
            step_size_post (float): Step size for Langevin dynamics in the posterior.
            ckpt (str): Checkpoint identifier for loading a trained model.

        Returns:
            np.ndarray: Generated samples from the prior distribution.
            np.ndarray: Generated samples from the posterior distribution.
        """
        data = kwargs['data']
        num_gen_samples = kwargs['num_gen_samples']
        n_iter = kwargs['n_iter']
        num_steps_post = kwargs['num_steps_post']
        step_size_post = kwargs['step_size_post']
        ckpt = kwargs['ckpt']

        dir_name = f'../{self.dataset}/{self.sampler}_{self.likelihood}/{num_steps_post}_{step_size_post}_lrGMM_{self.lrGMM}_lrG_{self.lrG}'
        ckpt_path = os.path.join(dir_name, f'ckpt/{ckpt}_model.pth')
        if not os.path.exists(ckpt_path):
            makedir(dir_name)
            logger = setup_logging('job0', dir_name, console=True)
            parameters = ['gmm', num_steps_post, step_size_post]
            logger.info(f'num_steps_prior, step_size_prior, num_steps_post, sampler {parameters}')
            self.train(data=data, n_iter=n_iter, batch_size=num_gen_samples, num_steps_post=num_steps_post, step_size_post=step_size_post)

        self.GMM, self.G = load_model(dir_name, f'{ckpt}_model', self.GMM, self.G)

        data_dir = os.path.join(dir_name, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        x = sample_p_data(data, num_gen_samples).squeeze()
        # x = self.normalize(x.squeeze())
        prior_final, means, lower_cholesky, weights = self.GMM(x.shape[0])
        self.testing = True
        self.plot = False
        post_final = langevin(x=x, 
                              z=prior_final, 
                              means=means, 
                              lower_cholesky=lower_cholesky, 
                              weights=weights, 
                              time=self.time, 
                              step_size=step_size_post, 
                              num_steps=num_steps_post, 
                              model=self.G, 
                              pushforward=self.pushforward, 
                              log_likelihood_sigma=self.log_likelihood_sigma, 
                              plot=self.plot,
                              testing=self.testing)[0]

        with torch.no_grad():
            output_prior = self.G(prior_final, self.time).squeeze()
            output_posterior = self.G(post_final, self.time).squeeze()

        np.save(os.path.join(data_dir, 'generated_priors.npy'), output_prior.cpu().numpy())
        np.save(os.path.join(data_dir, 'generated_posteriors.npy'), output_posterior.cpu().numpy())

        return output_prior, output_posterior