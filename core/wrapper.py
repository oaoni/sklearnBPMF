import matplotlib.pyplot as plt
import numpy as np
import smurff
import os
from sklearn.base import BaseEstimator

from sklearnBPMF.core import corr_metric


class Wrapper(BaseEstimator):
    """Scikit learn wrapper for matrix completion models."""

    def __init__(self, prior, num_latent, burnin, num_samples,
                 verbose, checkpoint_freq, save_freq, save_name,
                 num_threads, report_freq, metric_mode):

        self.prior = prior
        self.num_latent = num_latent
        self.burnin = burnin
        self.num_samples = num_samples
        self.verbose = verbose
        self.checkpoint_freq = checkpoint_freq
        self.save_freq = save_freq
        self.save_name = save_name + '.hdf5'
        self.num_threads = num_threads
        self.report_freq = report_freq
        self.metric_mode = metric_mode

        self.train_rmse = None
        self.test_corr = None
        self.sample_iter = None
        self.train_dict = None

    def fit(self, X_train, X_test, X_side, verbose=False, make_plot=True):
        # Initialize the training session method
        self.addData(X_train, X_test, X_side)

        # Train the model with the observed data
        while self.train_step():
            pass

        #self.store_metrics(self.num_samples)
        # self.train_rmse = self.train_dict['train_rmse'] # Deprecated in next version
        # self.test_rmse = self.train_dict['rmse_avg'] # Deprecated in next version
        # self.test_corr = self.train_dict['test_corr'] # Deprecated in next version

        if (verbose) and ((self.metric_mode == 1) or (self.metric_mode == 2)):
            print('Final test correlation is: {}'.format(self.train_dict['test_corr']))

        if (make_plot) and ((self.metric_mode == 1) or (self.metric_mode == 2)):
            self._makePlots(self.train_dict['pred_avg'], self.train_dict['pred_std'], X_test, self.train_dict['test_corr'])

        #Store training set predictions
        trainAvg, trainStd, trainCoord = self.predictTrain(return_std=True)
        self.train_dict['train_avg'] = trainAvg
        self.train_dict['train_std'] = trainStd
        self.train_dict['train_coord'] = trainCoord

        return self

    def addData(self, X_train, X_test, X_side):

        self._makeSession()
        self._makeModel(X_train, X_test, X_side)
        self.trainSession.init()

        self.X_train = X_train
        self.X_test = X_test

    def train_step(self):

        self.trainSession.step()

        self.sample_iter = self.trainSession.getStatus().iter

        if self.sample_iter % self.report_freq == 0:
            self.store_metrics(self.sample_iter, self.metric_mode)

        return self.sample_iter <= self.num_samples

    def store_metrics(self, sample_iter, metric_mode=0):

        macauStatus = self.trainSession.getStatus()

        if (metric_mode == 0): # Low memory mode
            # #Assign current training metrics w/o test predictions
            self.train_dict = dict(sample_iter = self.sample_iter,
                                   train_rmse = macauStatus.train_rmse,
                                   rmse_avg = macauStatus.rmse_avg,
                                   rmse_lsample = macauStatus.rmse_1sample)

        elif (metric_mode == 1) and (self.sample_iter != self.num_samples): # Low memory mode, but final high memory
            # #Assign current training metrics w/o test predictions
            self.train_dict = dict(sample_iter = self.sample_iter,
                                   train_rmse = macauStatus.train_rmse,
                                   rmse_avg = macauStatus.rmse_avg,
                                   rmse_lsample = macauStatus.rmse_1sample)

        else:
            # Get test predictions
            predAvg, predStd, predCoord = self.predict(return_std=True)
            testCorr = corr_metric(predAvg, self.X_test.data)

            #Assign current training metrics
            self.train_dict = dict(sample_iter = sample_iter,
                                   test_corr = testCorr,
                                   train_rmse = macauStatus.train_rmse,
                                   rmse_avg = macauStatus.rmse_avg,
                                   rmse_lsample = macauStatus.rmse_1sample,
                                   pred_avg = predAvg,
                                   pred_std = predStd,
                                   pred_coord = predCoord)

    def predict(self, return_std=False):
        # Return predicted unobserved values, does not require test data,
        # todo: add transform method for making predictions with arbitrary side
        # information and interactions

        # Get test predictions
        predictions = self.trainSession.getTestPredictions()
        predAvg = np.array([p.pred_avg for p in predictions])
        predCoord = [p.coords for p in predictions]

        if return_std:
            predVar = np.array([np.var(p.pred_all) for p in predictions])
            predStd = np.array([np.sqrt(p) for p in predVar])
            return predAvg, predStd, predCoord
        else:
            return predAvg, predCoord

    def predictTrain(self, return_std=False):
        # Return predicted observed values

        #Get train predictions
        predictor = self.trainSession.makePredictSession()
        p_all = predictor.predict_all()
        p_all_avg = sum(p_all) / len(p_all)
        predAvg = p_all_avg[self.train_coords[0],self.train_coords[1]]
        predCoord = list(zip(*self.train_coords))

        if return_std:
            predVar = np.var([p[self.train_coords[0], self.train_coords[1]] for p in p_all],axis=0)
            predStd = np.square(predVar)
            return predAvg, predStd, predCoord
        else:
            return predAvg, predCoord

    def _makePlots(self, predAvg, predStd, X_test, testCorr, saveplot=True):

        fig, ax = plt.subplots(3, 2, figsize=(15, 20))
        ax[0, 0].scatter(X_test.data, predAvg, edgecolors=(0, 0, 0))
        ax[0, 0].plot([X_test.data.min(), X_test.data.max()], [predAvg.min(), predAvg.max()], 'k--',
                      lw=4)
        ax[0, 0].set_xlabel('Measured')
        ax[0, 0].set_ylabel('Predicted')
        ax[0, 0].set_title('Measured vs Avg. Prediction')

        ax[0, 1].scatter(predStd, predAvg, edgecolors=(0, 0, 0))
        ax[0, 1].set_xlabel('Standard Deviation')
        ax[0, 1].set_ylabel('Predicted')
        ax[0, 1].set_title('Stdev. vs Prediction')

        x_ax = np.arange(len(X_test.data))
        sort_vals = np.argsort(X_test.data)
        ax[1, 0].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[1, 0].plot(x_ax, predAvg[sort_vals], 'rx', alpha=0.5, label='predicted')
        ax[1, 0].set_title('Sorted and overlayed measured and predicted values')
        ax[1, 0].legend()

        ax[1, 1].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[1, 1].plot(x_ax, predStd[sort_vals], 'r', label='stdev.')
        ax[1, 1].set_title('Sorted and overlayed measured stdev values')
        ax[1, 1].legend()

        ax[2, 0].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 0].fill_between(x_ax, predAvg[sort_vals] - predStd[sort_vals],
                              predAvg[sort_vals] + predStd[sort_vals],
                              alpha=0.5, label="std")
        ax[2, 0].set_title('predicted stdev. relative to predicted value')
        ax[2, 0].legend()

        ax[2, 1].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 1].fill_between(x_ax, X_test.data[sort_vals] - predStd[sort_vals],
                              X_test.data[sort_vals] + predStd[sort_vals],
                              alpha=0.5, label='std')
        ax[2, 1].set_title('predicted stdev. relative to measured value')
        ax[2, 1].legend()

        fig.tight_layout()
        fig.suptitle('{} - NSAMPLES: {} NUM_LATENT: {} SIDE_NOISE: {}\n Metrics - Corr: {:.5f}'. \
                     format(self.trainSession.getSaveName().split('.')[0],
                            self.num_samples, self.num_latent, self.side_noise, testCorr))

        fig.subplots_adjust(top=0.90)
        if saveplot:
            fig.savefig('figure_log.png')
            print("Saved figure to current working directory..")

        return self

    def _makeSession(self):
        #if exists, replace
        if os.path.isfile(self.save_name):
            os.remove(self.save_name)

        self.trainSession = smurff.TrainSession(
            priors=[self.prior, self.prior],
            num_latent=self.num_latent,
            burnin=self.burnin,
            nsamples=self.num_samples,
            verbose=self.verbose,
            checkpoint_freq=self.checkpoint_freq,
            save_freq=self.save_freq,
            num_threads=self.num_threads,
            save_name=self.save_name)

        return self