import smurff
import copy
import numpy as np

class BayesianRegression():
    def __init__(self,alpha_,sigma_,tol=1e-3,max_iters=0,bias_=True):

        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.tol = tol
        self.max_iters = max_iters
        self.bias_ = bias_

        self.cov_ = None
        self.mu_ = None

    def fit(self,X,y):

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

    def transform(self,X):

        y_star = X @ self.mu_

        if self.bias_:
            y_pred = y_star[:,1:X.shape[1]] + y_star[:,1:X.shape[1]].T
        else:
            y_pred = y_star + y_star.T

        return y_pred

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
