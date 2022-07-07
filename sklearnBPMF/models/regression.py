import smurff
import copy
import numpy as np
import pandas as pd
from sklearnBPMF.data.utils import add_bias, verify_ndarray
from sklearnBPMF.core.metrics import score_completion

class BayesianRegression:
    def __init__(self,alpha_init,sigma_init,model='collective',tol=1e-3,max_iters=0,bias=True,
                 bias_both_dim=False,M=None,S_train=None,S_test=None,k_metrics=True,k=20):

        self.alpha = alpha_init
        self.sigma = sigma_init
        self.model = model
        self.tol = tol
        self.max_iters = max_iters
        self.bias = bias
        self.bias_both_dim = bias_both_dim
        self.k_metrics = k_metrics
        self.k = k

        self.cov_ = None
        self.mu_ = None
        self.side = None

        self.M = M
        self.S_train = S_train
        self.S_test = S_test

    def fit(self,X,y,side=None,X_test=None):

        self.X_train = X
        self.y = y
        self.X_test = X_test

        X,y = self.format_data(X,y=y,side=side)

        # Initial prediction of the weight posterior mean and covariance
        self.cov_, self.mu_ = self.weight_posterior(X,y,self.alpha,self.sigma)

        # Compute uncertainty
        self.uncertainty = self.compute_variance(X)

        # Store performance metrics
        self.store_metrics(self.X_train)

        for i in range(self.max_iters):
                # Compute the log likelihood
        #         log_likes += [log_likelihood(X,y,alpha_,sigma_,mu_)]

            alpha_old = copy.copy(self.alpha)
            sigma_old = copy.copy(self.sigma)

            # Update params
            self.alpha, self.sigma = self.update_params(X,y,self.alpha,self.cov_, self.mu_)
            self.cov_, self.mu_ = self.weight_posterior(X,y,self.alpha,self.sigma)

            # Compute uncertainty
            self.uncertainty = compute_variance(X)

            # Store performance metrics
            self.store_metrics(self.X_train)

            # Check for convergence
        #     converg = sum(abs(alpha_old - alpha_))
            converg = abs(sigma_old - self.sigma)
            if converg < self.tol:
                print("Convergence after ", str(i), " iterations")
                break

    def compute_variance(self, X):

        variance = ((X @ self.cov_) @ (X.T)) + self.sigma

        return pd.DataFrame(variance + variance.T)


    def transform(self,X):

        X = verify_ndarray(X)

        # Format bias and perform transformation
        if self.bias:
            X = add_bias(X,both_dims=self.bias_both_dim)
            if self.bias_both_dim:
                y_star = (X @ self.mu_)[1:,1:]
            else:
                y_star = (X @ self.mu_)[:,1:]
        else:
            y_star = X @ self.mu_

        y_pred = y_star + y_star.T

        return pd.DataFrame(y_pred)

    def predict(self,S='test',return_std=False):

        if S == 'test':
            S = self.S_test
        elif S == 'train':
            S = self.S_train

        pred = (self.Xhat * S.replace(0,np.nan).values).stack()
        avg = list(pred.values.astype(np.float32))
        coord = list(pred.index.values)
        std = list(self.uncertainty.stack()[coord].values.astype(np.float32))

        if return_std:
            return avg, std, coord
        else:
            return avg, coord


    def store_metrics(self,X):

        self.Xhat = self.transform(X)

        test_dict = score_completion(self.M,self.Xhat,self.S_test,'test',k_metrics=self.k_metrics,k=self.k)
        train_dict = score_completion(self.M,self.Xhat,self.S_train,'train',k_metrics=self.k_metrics,k=self.k)

        pred_avg, pred_std, pred_coord = self.predict(return_std=True)
        train_avg, train_std, train_coord = self.predict(S='train',return_std=True)

        #Assign current training metrics
        self.train_dict = dict(pred_avg = pred_avg,
                               pred_std = pred_std,
                               pred_coord = pred_coord,
                               train_avg = train_avg,
                               train_std = train_std,
                               train_coord = train_coord,
                               **test_dict,
                               **train_dict)

    def format_data(self,X,y=None,side=None):

        if y is None:
            y = copy.copy(X)

        X,y,side = verify_ndarray(X,y,side)

        # Format data with side information
        if self.model == 'collective':
            self.side = side
            X = np.concatenate([X,side])
            if isinstance(y,np.ndarray):
                y = np.concatenate([y,side])

        # Format bias
        if self.bias:
            X = add_bias(X,both_dims=self.bias_both_dim)
            if isinstance(y,np.ndarray):
                y = add_bias(y,both_dims=self.bias_both_dim)

        # Format alpha initilization as a diagonal matrix
        self.alpha = np.diag(self.alpha*np.ones(X.shape[1]))

        return X,y

    # Weight posterior
    def weight_posterior(self,X,y,alpha,sigma):

        cov = np.linalg.pinv((sigma**-1)*(X.T @ X) + alpha)
        mu = (sigma**-1)*((cov @ X.T) @ y)

        return cov, mu

    def update_params(self, X,y,alpha,cov,mu):

        # gamma = (alpha*cov)
        gamma =  np.diag(np.ones(cov.shape[0])) - (alpha*cov)
        N = X.shape[0]

        alpha_new = gamma/(mu**2)
        sigma_new = ((y - (X @ mu))**2).sum()/(N - gamma.sum())

        return np.nan_to_num(alpha_new), np.nan_to_num(sigma_new)

    def log_likelihood(self, X,y,alpha,sigma,mu):

        return (sigma**(-1/2)*((y - (X @ mu))**2).sum() + mu.T @ alpha @ mu)
