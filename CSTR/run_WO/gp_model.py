import math
import torch
import gpytorch
import numpy as np
import os
import matplotlib.pyplot as plt

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood='Gaussian', kernel='RBF'):
        if likelihood == 'Gaussian':
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # convert to torch if provided numpy array
        if type(train_x) == np.ndarray:
            train_x = torch.from_numpy(train_x).double()
        if type(train_y) == np.ndarray:
            train_y = torch.from_numpy(train_y).double()

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            # self.mean_module.register_constraint('constant', gpytorch.constraints.Interval(425, 500))
            # self.covar_module.base_kernel.register_constraint('raw_lengthscale', gpytorch.constraints.Interval(1, 500.0))
            # self.likelihood.noise_covar.register_constraint('raw_noise', gpytorch.constraints.Interval(1.0, 100.0))
            #self.covar_module.register_constraint('raw_outputscale', gpytorch.constraints.Interval(1.0, 1000.0))

        if kernel == '2poly':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.polynomial_kernel(2))
        if kernel == '2poly+RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.polynomial_kernel(2))
        if kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        if kernel == 'Matern+linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel() + gpytorch.kernels.linear_kernel())
        self.double()


    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).double()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BOGPModels:
    """
    Implement the specific gp models used for Bayesian optimization.
    """
    def __init__(self, train_X, train_obj, train_constr, x0_arr, obj_arr, constr_arr):
        self.train_X = train_X
        self.train_obj = train_obj
        self.train_constr = train_constr

        self.x0_arr = x0_arr
        self.safe_obj_arr = obj_arr
        self.safe_constr_arr = constr_arr
        num_safe_points, num_constrs = constr_arr.shape
        self.num_constrs = num_constrs

        self.obj_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_obj_torch = ExactGPModel(self.train_X, self.train_obj[:, 0], self.obj_likelihood)

        self.infer_obj_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.infer_gp_obj_torch = ExactGPModel(self.x0_arr, self.safe_obj_arr[:, 0], self.infer_obj_likelihood)

        self.constr_likelihood_list = []
        self.gp_constr_torch_list = []

        self.infer_constr_likelihood_list = []
        self.infer_gp_constr_torch_list = []

        for i in range(self.num_constrs):
            constr_obj = self.train_constr[:, i]

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp_constr_torch = ExactGPModel(self.train_X, constr_obj, likelihood)

            infer_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            infer_gp_constr_torch = ExactGPModel(self.x0_arr, self.safe_constr_arr[:, i], infer_likelihood)
            self.constr_likelihood_list.append(likelihood)
            self.gp_constr_torch_list.append(gp_constr_torch)

            self.infer_constr_likelihood_list.append(infer_likelihood)
            self.infer_gp_constr_torch_list.append(infer_gp_constr_torch)


        self.train_torch_gp()

        self.infer_gp_obj_torch.initialize(**dict(self.gp_obj_torch.named_hyperparameters()))
        self.infer_obj_likelihood.initialize(**dict(self.obj_likelihood.named_hyperparameters()))
        for k in range(self.num_constrs):
            self.infer_gp_constr_torch_list[k].initialize(**dict(self.gp_constr_torch_list[k].named_hyperparameters()))
            self.infer_constr_likelihood_list[k].initialize(**dict(self.constr_likelihood_list[k].named_hyperparameters()))


    def train_torch_gp(self, verbal=False):
        # train the torch gp
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        smoke_test = ('CI' in os.environ)
        training_iter = 2 if smoke_test else 5000

        self.gp_obj_torch.train()
        self.obj_likelihood.train()

        optimizer = torch.optim.Adam(self.gp_obj_torch.parameters(), lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.obj_likelihood, self.gp_obj_torch)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.gp_obj_torch(torch.from_numpy(self.train_X).double())
            loss = -mll(output, torch.from_numpy(self.train_obj[:, 0]).double())
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f lengthscale %.3f noise: %.3f for objective gp.' % (
            #     i + 1, training_iter, loss.item(),
            #     self.gp_obj_torch.covar_module.base_kernel.lengthscale.item(),
            #     self.gp_obj_torch.likelihood.noise.item()
            # ))
            optimizer.step()

        for k in range(self.num_constrs):
            self.gp_constr_torch_list[k].train()
            self.constr_likelihood_list[k].train()

            optimizer = torch.optim.Adam(self.gp_constr_torch_list[k].parameters(), lr=0.01)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.constr_likelihood_list[k], self.gp_constr_torch_list[k])

            for i in range(training_iter):
                optimizer.zero_grad()
                output = self.gp_constr_torch_list[k](torch.from_numpy(self.train_X).double())
                loss = -mll(output, torch.from_numpy(self.train_constr[:, k]).double())
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f lengthscale %.3f noise: %.3f for constraint %d gp.' % (
                #     i + 1, training_iter, loss.item(),
                #     self.gp_constr_torch_list[k].covar_module.base_kernel.lengthscale.item(),
                #     self.gp_constr_torch_list[k].likelihood.noise.item(),
                #     k + 1
                # ))
                optimizer.step()

    def train_torch_infer_gp(self, verbal=False, training_iter=20):
        # train the torch gp
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        smoke_test = ('CI' in os.environ)
        training_iter = 2 if smoke_test else 100

        self.infer_gp_obj_torch.train()
        self.infer_obj_likelihood.train()

        optimizer = torch.optim.Adam(self.infer_gp_obj_torch.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.infer_obj_likelihood, self.infer_gp_obj_torch)

        for i in range(training_iter):
            optimizer.zero_grad()
            train_X = self.infer_gp_obj_torch.train_inputs
            train_Y = self.infer_gp_obj_torch.train_targets
            output = self.infer_gp_obj_torch(train_X[0])
            loss = -mll(output, train_Y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f lengthscale %.3f noise: %.3f for objective gp.' % (
            #     i + 1, training_iter, loss.item(),
            #     self.infer_gp_obj_torch.covar_module.base_kernel.lengthscale.item(),
            #     self.infer_gp_obj_torch.likelihood.noise.item()
            # ))
            optimizer.step()

        for k in range(self.num_constrs):
            self.infer_gp_constr_torch_list[k].train()
            self.infer_constr_likelihood_list[k].train()

            optimizer = torch.optim.Adam(self.infer_gp_constr_torch_list[k].parameters(), lr=0.1)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.infer_constr_likelihood_list[k], self.infer_gp_constr_torch_list[k])

            for i in range(training_iter):
                optimizer.zero_grad()
                train_X = self.infer_gp_constr_torch_list[k].train_inputs
                train_Y = self.infer_gp_constr_torch_list[k].train_targets
                output = self.infer_gp_constr_torch_list[k](train_X[0])
                loss = -mll(output, train_Y)
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f lengthscale %.3f noise: %.3f for constraint %d gp.' % (
                #     i + 1, training_iter, loss.item(),
                #     self.infer_gp_constr_torch_list[k].covar_module.base_kernel.lengthscale.item(),
                #     self.infer_gp_constr_torch_list[k].likelihood.noise.item(),
                #     k + 1
                # ))
                optimizer.step()

    def append_new_data(self, gp,  x, y, normalize):
        train_inputs = gp.train_inputs
        train_targets = gp.train_targets

        new_train_inputs = torch.cat((train_inputs[0], torch.from_numpy(x).double()))
        new_train_targets = torch.cat((train_targets, torch.from_numpy(y).double()))
        return new_train_inputs, new_train_targets

    def set_new_datas(self, new_X, new_obj, new_constrs_list, update_hyper_params=False):
        self.infer_gp_obj_torch.set_train_data(torch.from_numpy(new_X).double(), torch.from_numpy(new_obj[:, 0]).double(), strict=False)
        for i in range(self.num_constrs):
            self.infer_gp_constr_torch_list[i].set_train_data(torch.from_numpy(new_X).double(), torch.from_numpy(new_constrs_list[i][:, 0]).double(), strict=False)

        if update_hyper_params:
            self.train_torch_infer_gp(verbal=True)

    def add_new_data_point(self, x, y_obj, constr_val, update_hyper_params=False, normalize=True):
        # update obj gp
        new_obj_inputs, new_obj_targets = self.append_new_data(self.infer_gp_obj_torch, x, y_obj[:,0], normalize)
        self.infer_gp_obj_torch.set_train_data(new_obj_inputs, new_obj_targets, strict=False)
        for i in range(self.num_constrs):
            new_inputs, new_targets = self.append_new_data(self.infer_gp_constr_torch_list[i], x, constr_val[:, i], normalize)
            self.infer_gp_constr_torch_list[i].set_train_data(new_inputs, new_targets, strict=False)

        if update_hyper_params:
            self.train_torch_infer_gp()

    def plot_one_gp(self, model, test_x, likelihood, dim=1):
        num_data, x_dim = test_x.shape

        if type(test_x) == np.ndarray:
            test_x = torch.from_numpy(test_x).double()

        model.eval()
        likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))

        if dim == 1:
            with torch.no_grad():
                # Initialize plot
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                # Plot training data as black stars
                train_x = model.train_inputs
                train_y = model.train_targets
                ax.plot(train_x[0].numpy(), train_y.numpy(), 'k*')
                # Plot predictive means as blue line
                ax.plot(test_x.numpy()[:, 0], observed_pred.mean.numpy(), 'b')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy()[:, 0], lower.numpy(), upper.numpy(), alpha=0.5)
                ax.legend(['Observed Data', 'Mean', 'Confidence'])

    def plot_gps(self, x):
        num_data, x_dim = x.shape
        if x_dim == 1:
            self.plot_one_gp(self.infer_gp_obj_torch, x, self.infer_obj_likelihood)
            for k in range(self.num_constrs):
                self.plot_one_gp(self.infer_gp_constr_torch_list[k], x, self.infer_constr_likelihood_list[k])





