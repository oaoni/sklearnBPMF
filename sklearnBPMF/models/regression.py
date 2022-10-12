import smurff
import copy
import numpy as np
import pandas as pd
from sklearnBPMF.data.utils import add_bias, verify_ndarray, verify_pdframe
from sklearnBPMF.core.metrics import score_completion

class BayesianRegression:
    def __init__(self,alpha_init,sigma_init,model='collective',tol=1e-3,max_iters=0,bias=True,
                 bias_both_dim=False,k_metrics=True,k=20,transpose_variance=True):

        self.alpha = alpha_init
        self.sigma = sigma_init
        self.model = model
        self.tol = tol
        self.max_iters = max_iters
        self.bias = bias
        self.bias_both_dim = bias_both_dim
        self.k_metrics = k_metrics
        self.k = k
        self.transpose_variance = transpose_variance

        self.cov_ = None
        self.mu_ = None
        self.side = None
        self.S_train = None
        self.S_test = None

    def fit(self,X_train,X_side=None,X_test=None,y=None,M=None):

        #Store training and evaluation data
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.M = M

        # Produce masks
        self.S_train = verify_pdframe((X_train != 0)*1)
        self.S_test = verify_pdframe((X_test != 0)*1)

        # Format data for training
        X,y = self.format_data(X_train,y=y,side=X_side)

        # Initial prediction of the weight posterior mean and covariance
        self.cov_, self.mu_ = self.weight_posterior(X,y,self.alpha,self.sigma)

        # Compute uncertainty
        self.uncertainty = self.compute_variance(X,transpose=self.transpose_variance)

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
            self.uncertainty = compute_variance(X,transpose=self.transpose_variance)

            # Store performance metrics
            self.store_metrics(self.X_train)

            # Check for convergence
        #     converg = sum(abs(alpha_old - alpha_))
            converg = abs(sigma_old - self.sigma)
            if converg < self.tol:
                print("Convergence after ", str(i), " iterations")
                break

    def compute_variance(self, X, transpose=True):

        variance = ((X @ self.cov_) @ (X.T)) + self.sigma

        # Format bias and perform transformation
        if self.bias_both_dim:
            variance = variance[1:,1:]

        if transpose:
            variance += variance.T

        uncertainty = (variance - variance.min()) / (variance.max() - variance.min())

        return pd.DataFrame(uncertainty)


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
        cond_number = np.linalg.cond(verify_ndarray(X) + np.eye(X.shape[0]))
        cond_number_mask = np.linalg.cond(self.S_train + np.eye(X.shape[0]))

        pred_avg, pred_std, pred_coord = self.predict(return_std=True)
        train_avg, train_std, train_coord = self.predict(S='train',return_std=True)

        #Assign current training metrics
        self.train_scores = train_dict
        self.test_scores = test_dict
        self.train_dict = dict(pred_avg = pred_avg,
                               pred_std = pred_std,
                               pred_coord = pred_coord,
                               train_avg = train_avg,
                               train_std = train_std,
                               train_coord = train_coord,
                               cond_number = cond_number,
                               cond_number_mask = cond_number_mask,
                               **test_dict,
                               **train_dict)

    def format_data(self,X,y=None,side=None):

        if y is None:
            y = copy.copy(X)
            self.y = y

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
        if not isinstance(self.alpha,np.ndarray):
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
