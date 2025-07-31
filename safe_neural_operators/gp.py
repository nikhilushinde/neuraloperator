"""
File for Gaussian Process (GP) related functionalities in the safe_neural_operators package.
"""

# 1. GP Regression Module 

import numpy as np
import GPy
from copy import deepcopy

import jax 
import jax.numpy as jnp 
from jax import vmap

class GPWrapper:
    def __init__(self, X_init, Y_init, kernel=None, noise_var=1e-6, optimize=True):
        """
        Initialize the GP model with data and a kernel.

        Parameters:
        - X_init: Initial input data (n_samples, input_dim)
        - Y_init: Initial output data (n_samples, 1)
        - kernel: GPy kernel instance, optional
        - noise_var: Gaussian noise variance
        """

        self.X = np.array(X_init)
        self.Y = np.array(Y_init)
        
        input_dim = self.X.shape[1]
        self.kernel = kernel if kernel else GPy.kern.RBF(input_dim=input_dim)
        self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel, noise_var=noise_var)
        self.model.Gaussian_noise.variance = noise_var
        if optimize:
            self.model.optimize()
        
        self.update_jax_predictor()

    def add_sample(self, x_new, y_new, reoptimize=True):
        """
        Add new sample(s) and retrain the GP.
        """
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        self.X = np.vstack((self.X, x_new))
        self.Y = np.vstack((self.Y, y_new))

        print("Adding new sample to GP model:", self.X.shape, self.Y.shape)

        self.model.set_XY(self.X, self.Y)
        if reoptimize:
            self.model.optimize()

        self.update_jax_predictor()

    def predict(self, X_test):
        """
        Predict the mean and variance at new input points.
        """
        X_test = np.atleast_2d(X_test)
        mu, var = self.model.predict(X_test)
        return mu, var
    
    def predict_mean_gradient(self, x_test):
        """
        Compute the gradient of the GP posterior mean at x_test.
        Reference: https://mlg.eng.cam.ac.uk/pub/pdf/Mch14.pdf Chapter 2.7
        
        Parameters:
        - x_test: (n_test, input_dim) array of query points
        
        Returns:
        - grad_mu: (n_test, input_dim) array of gradient vectors
        """
        x_test = np.atleast_2d(x_test)
        X_train = self.X
        Y_train = self.Y
        kernel = self.model.kern

        lengthscale = kernel.lengthscale.values[0]
        variance = kernel.variance.values[0]
        noise_var = self.model.Gaussian_noise.variance[0]

        K = kernel.K(X_train) + np.eye(len(X_train)) * noise_var
        # alpha = np.linalg.solve(K, Y_train)
        alpha = self.model.posterior.woodbury_vector

        grad_mu = np.zeros_like(x_test)

        for i, x_star in enumerate(x_test):
            k_vec = kernel.K(X_train, x_star[None, :])  # (n_train, 1)
            x_diff = (X_train - x_star) / (lengthscale**2)  # (n_train, input_dim)
            weighted = k_vec * x_diff  # element-wise
            grad_mu[i] = (weighted.T @ alpha).ravel()

        return grad_mu

    def predict_mean_var_gradient(self, x_test):
        """
        Compute the gradient of the GP posterior mean and var at x_test.
        Reference: https://mlg.eng.cam.ac.uk/pub/pdf/Mch14.pdf Chapter 2.7
        
        Parameters:
        - x_test: (n_test, input_dim) array of query points
        
        Returns:
        - grad_mean: (n_test, input_dim) array of gradient vectors
        - grad_var: (n_test, input_dim, input_dim) array of gradient matrices
        """

        x_test = np.atleast_2d(x_test)
        X_train = self.X
        Y_train = self.Y
        kernel = self.model.kern

        lengthscale = kernel.lengthscale.values[0]
        variance = kernel.variance.values[0]
        noise_var = self.model.Gaussian_noise.variance[0]

        # K = kernel.K(X_train) + np.eye(len(X_train)) * noise_var
        # alpha = np.linalg.solve(K, Y_train)
        alpha = self.model.posterior.woodbury_vector

        grad_mu = np.zeros_like(x_test)
        grad_var = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1]))

        for i, x_star in enumerate(x_test):
            k_vec = kernel.K(X_train, x_star[None, :])  # (n_train, 1)
            x_diff = (X_train - x_star) / (lengthscale**2)  # (n_train, input_dim)
            
            # 1st derivatives of kernel function: dk(X_train, x_star) / dx_star and dk(x_star, X_train) / dx_star 
            dkXtrainxtest_dxtest = x_diff * k_vec  # element-wise
            dkxtestXtrain_dxtest = dkXtrainxtest_dxtest.T  

            grad_mu[i] = (dkxtestXtrain_dxtest @ alpha).ravel()



            # 2nd derivatives of kernel function: d^2 k(x_test, x_test) 
            dkxtestxtest_dxtestxtest = np.eye(x_test.shape[1]) * (variance/lengthscale**2)  # diagonal matrix

            grad_var[i] = dkxtestxtest_dxtestxtest - (dkxtestXtrain_dxtest @ self.model.posterior.woodbury_inv @ dkXtrainxtest_dxtest)
            # print(f"dkxtestxtest_dxtestxtest : {dkxtestxtest_dxtestxtest}")
            # print(f"other shape: {(dkxtestXtrain_dxtest @ self.model.posterior.woodbury_inv @ dkXtrainxtest_dxtest)}")
            # print()

        return grad_mu, grad_var
    
    def predict_mean_var_gradient_vectorized(self, x_test):
        """
        Vectorized version to compute the gradient of the GP posterior mean and var at x_test.
        Reference: https://mlg.eng.cam.ac.uk/pub/pdf/Mch14.pdf Chapter 2.7
        
        Parameters:
        - x_test: (n_test, input_dim) array of query points
        
        Returns:
        - grad_mean: (n_test, input_dim) array of gradient vectors
        - grad_var: (n_test, input_dim, input_dim) array of gradient matrices
        """
        x_test = np.atleast_2d(x_test)
        X_train = self.X
        Y_train = self.Y
        kernel = self.model.kern

        lengthscale = kernel.lengthscale.values[0]
        variance = kernel.variance.values[0]
        noise_var = self.model.Gaussian_noise.variance[0]
        
        alpha = self.model.posterior.woodbury_vector # (n_train, 1)

        grad_mu = np.zeros_like(x_test)
        grad_var = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1]))

        
        # Vectorized computation 
        batch_shape = x_test.shape
        n_batch_dims = len(batch_shape)
        n1 = x_test.shape[0]
        n2 = X_train.shape[0]
        d = x_test.shape[1]


        # Batch mean computation 
        k_vec = kernel.K(x_test, X_train) # (n_test, n_train)
        x_diff = - (x_test[:, None, :]  - X_train[None, :, :]) / (lengthscale**2) # (n_test, n_train, input_dim)
        dkXtrainxtest_dxtest = x_diff * k_vec[:, :, None]  # element-wise (n_test, n_train, input_dim)
        dkxtestXtrain_dxtest = dkXtrainxtest_dxtest.transpose(0, 2, 1)   # element-wise (n_test, input_dim, n_train)
        grad_mu = (dkxtestXtrain_dxtest @ alpha).reshape(n1, d)

        # Batch variance computation 
        # TODO: NEEDS TO BE FINISHED! 

        for i, x_star in enumerate(x_test):
            k_vec = kernel.K(X_train, x_star[None, :])  # (n_train, 1)
            x_diff = (X_train - x_star) / (lengthscale**2)  # (n_train, input_dim)
            
            # 1st derivatives of kernel function: dk(X_train, x_star) / dx_star and dk(x_star, X_train) / dx_star 
            dkXtrainxtest_dxtest = x_diff * k_vec  # element-wise
            dkxtestXtrain_dxtest = dkXtrainxtest_dxtest.T  

            grad_mu[i] = (dkxtestXtrain_dxtest @ alpha).ravel()



            # 2nd derivatives of kernel function: d^2 k(x_test, x_test) 
            dkxtestxtest_dxtestxtest = np.eye(x_test.shape[1]) * (variance/lengthscale**2)  # diagonal matrix

            grad_var[i] = dkxtestxtest_dxtestxtest - (dkxtestXtrain_dxtest @ self.model.posterior.woodbury_inv @ dkXtrainxtest_dxtest)
            # print(f"dkxtestxtest_dxtestxtest : {dkxtestxtest_dxtestxtest}")
            # print(f"other shape: {(dkxtestXtrain_dxtest @ self.model.posterior.woodbury_inv @ dkXtrainxtest_dxtest)}")
            # print()

        return grad_mu, grad_var
    
    def get_model(self):
        """
        Return the internal GPy model.
        """
        return self.model
    

    def update_jax_predictor(self): 
        """
        Create a predict function that fully uses jax
        """
        # Get Kernel Hyperparameters
        self.lengthscale_kern_jax = float(deepcopy(self.model.kern.lengthscale.values))
        self.variance_kern_jax = float(deepcopy(self.model.kern.variance.values))

        # Convert training data 
        self.X_train_jax = jnp.asarray(deepcopy(self.model.X))
        self.Y_train_jax = jnp.asarray(deepcopy(self.model.Y))
        self.alpha_jax = jnp.asarray(deepcopy(self.model.posterior.woodbury_vector))
        self.K_inv_jax = jnp.asarray(deepcopy(self.model.posterior.woodbury_inv))

    def rbf_kernel_jax(self, X1, X2): 
        """
        Compute full RBF kernel matrix between two sets of inputs:
        X1: (M, D), X2: (N, D) => returns (M, N)
        """
        # NOTE: Later add support for multiple lengthscales and variances
        X1_sq = jnp.sum(X1**2, axis=1, keepdims=True)  # (M, 1)
        X2_sq = jnp.sum(X2**2, axis=1, keepdims=True).T  # (1, N)
        sq_dists = X1_sq + X2_sq - 2 * jnp.dot(X1, X2.T)  # (M, N)
        return self.variance_kern_jax * jnp.exp(-0.5 * sq_dists / (self.lengthscale_kern_jax ** 2))  # (M, N)

    def predict_jax(self, x_test): 
        """
        Predict GP mean and variance at batched inputs xnew: (M, D)
        Returns:
            mean: (M, 1)
            variance: (M, 1)
        """
        # Compute kernel between test points and training data
        k_x_test = self.rbf_kernel_jax(jnp.asarray(x_test), self.X_train_jax)  # (M, N)

        # Mean: (M, 1) = (M, N) @ (N, 1)
        mean = k_x_test @ self.alpha_jax  # (M, 1)

        # Variance: diag(K_xx - K_xs @ K_inv @ K_xs.T)
        # K_xx is just a constant vector: variance (because RBF(x, x) = variance)
        # K_inv_K_xs_T = self.K_inv_jax @ k_x_test.T  # (N, M)
        # var = self.variance_kern_jax - jnp.sum(k_x_test * K_inv_K_xs_T.T, axis=1, keepdims=True)  
        var = self.variance_kern_jax - jnp.einsum("mi,ij,mj->m", k_x_test, self.K_inv_jax, k_x_test).reshape(-1, 1) # (M, 1)

        return mean, var

# 2. GP Regression Model -> image input on grid 
def gp_model_to_2dinput_image(gp_regression_model, grid_states): 
    """
    Get the 2D disturbance input image from the GP model: 
    NOTE: GP model is only trained on [x,y] -> disturbance value: doesn't care about xvel, yvel here
    As a result image is disturbance value over the x,y grid only
    """
    # grid_states: [x, y, xvel, yvel, 4] state: [x,y,xvel,yvel]

    # Get states corresponding to image grid 
    image_grid = grid_states[:, :, 0, 0, :]


    # Flatten so you can do batch 

    # Get the corresponding outputs 

    # Reshape into same grid size 

    # Output as Image
    raise NotImplementedError("TODO: implement")

