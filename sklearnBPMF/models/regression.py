import smurff
import copy
import numpy as np
from sklearnBPMF.data.utils import add_bias

class BayesianRegression:
    def __init__(self,alpha,sigma,model='collective',tol=1e-3,max_iters=0,bias=True, bias_both_dim=False):

        self.alpha_ = alpha
        self.sigma_ = sigma
        self.model = model
        self.tol = tol
        self.max_iters = max_iters
        self.bias = bias
        self.bias_both_dim = bias_both_dim

        self.cov_ = None
        self.mu_ = None
        self.side = None

    def fit(self,X,y,side=None):

        X,y = self.format_data(X,y,side)

        # Initial prediction of the weight posterior mean and covariance
        self.cov_, self.mu_ = self.weight_post(X,y,self.alpha_,self.sigma_)

        for i in range(self.max_iters):
                # Compute the log likelihood
        #         log_likes += [log_like(X,y,alpha_,sigma_,mu_)]

            alpha_old = copy.copy(self.alpha_)
            sigma_old = copy.copy(self.sigma_)

            self.alpha_, self.sigma_ = self.update_params(X,y,alpha_,cov_, mu_)
            self.cov_, self.mu_ = self.weight_post(X,self.alpha_,self.sigma_)

            # Check for convergence
        #     converg = sum(abs(alpha_old - alpha_))
            converg = abs(sigma_old - self.sigma_)
            if converg < tol:
                print("Convergence after ", str(i), " iterations")
                break

        # Compute uncertainty
        variance = ((X @ self.cov_) @ (X.T)) + self.sigma_
        self.variance = variance + variance.T

    def transform(self,X):

        X,_ = self.format_data(X,X,None)

        if self.bias:
            if self.bias_both_dim:
                y_star = (X @ self.mu_)[1:,1:]
            else:
                y_star = (X @ self.mu_)[,1:]
        else:
            y_star = X @ self.mu_

        y_pred = y_star + y_star.T

        return y_pred

    def format_data(self,X,y,side):

        # Format data with side information
        if self.model == 'noside':
            X_ = X
        elif self.model == 'collective':
            self.side = side
            X_ = pd.concat([X,side],axis=0)

        # Format bias
        if self.bias:
            X = self.add_bias(X,both_dims=self.bias_both_dim)
            y = self.add_bias(y,both_dims=self.bias_both_dim)

        return X,y

    # Weight posterior
    def weight_post(self, X,y,alpha,sigma):

        A = alpha

        cov = np.linalg.pinv((sigma**-1)*(X.T @ X) + A)
        mu = (sigma**-1)*((cov @ X.T) @ y)

        return cov, mu

    def update_params(self, X,y,alpha,cov,mu):

#         gamma = 1 - (np.diag(alpha)*np.diag(cov))
#         gamma =  np.diag(np.ones(cov.shape[0])) - (alpha*cov)
        gamma = (alpha*cov)
        N = X.shape[0]

        alpha_new = gamma/(mu**2)
        sigma_new = ((y - (X @ mu))**2).sum()/(N - gamma.sum())
        print(((y - (X @ mu))**2).sum(),(N - gamma.sum()))

        return np.nan_to_num(alpha_new), np.nan_to_num(sigma_new)

    def log_like(self, X,y,alpha,sigma,mu):

        A = alpha

        return (sigma**(-1/2)*((y - (X @ mu))**2).sum() + mu.T @ A @ mu)
