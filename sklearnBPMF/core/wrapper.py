import matplotlib.pyplot as plt
import numpy as np
import smurff
import os
import seaborn as sns
from sklearn.base import BaseEstimator
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import coo_matrix

from sklearnBPMF.core import corr_metric
from sklearnBPMF.data.utils import to_sparse


class Wrapper(BaseEstimator):
    """Scikit learn wrapper for matrix completion models."""

    def __init__(self,num_latent, burnin, num_samples,
                 verbose, checkpoint_freq, save_freq, save_name,
                 num_threads, report_freq, metric_mode, col_side,
                 keep_file):

        self.num_latent = num_latent
        self.burnin = burnin
        self.num_samples = num_samples
        self.verbose = verbose
        self.checkpoint_freq = checkpoint_freq
        self.save_freq = save_freq
        self.save_name = save_name + '.hdf5' if keep_file else save_name + '.tmp.hdf5'
        self.num_threads = num_threads
        self.report_freq = report_freq if report_freq else num_samples
        self.metric_mode = metric_mode
        self.col_side = col_side
        self.keep_file = keep_file

        self.train_rmse = None
        self.test_corr = None
        self.sample_iter = None
        self.train_dict = None

    def fit(self, X_train, X_test, X_side, verbose=False, make_plot=True, complete_matrix=None):
        # Initialize the training session method
        self.addData(X_train, X_test, X_side)

        # Train the model with the observed data
        while self.train_step():
            pass

        if self.metric_mode > 0:
            #Store training set predictions
            trainAvg, trainStd, trainCoord = self.predictTrain(return_std=True)
            self.train_dict['train_avg'] = trainAvg
            self.train_dict['train_std'] = trainStd
            self.train_dict['train_coord'] = trainCoord

        if (verbose) and ((self.metric_mode == 1) or (self.metric_mode == 2)):
            print('Final test correlation is: {}'.format(self.train_dict['test_corr']))

        if (make_plot) and ((self.metric_mode == 1) or (self.metric_mode == 2)):
            self._makePlots(self.train_dict, X_train, X_test, complete_matrix=complete_matrix)

        if not self.keep_file:
            print('Deleting temporary save file: {}'.format(self.save_name))
            os.remove(self.save_name)

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

        else: # High memory mode
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

    def _makePlots(self, train_dict, X_train, X_test, saveplot=True, complete_matrix=None):

        pred_avg = train_dict['pred_avg']
        pred_std = train_dict['pred_std']
        test_corr = train_dict['test_corr']
        rmse_avg = train_dict['rmse_avg']
        pred_coord = train_dict['pred_coord']
        train_avg = train_dict['train_avg']
        train_std = train_dict['train_std']
        train_coord = train_dict['train_coord']
        shape = X_test.shape
        n_train_examples = len(X_train.data)//2
        test_ratio = len(X_test.data)/(shape[0]**2)

        if complete_matrix != None:
            M = complete_matrix.tocsc()
        else:
            M = (X_train + X_test)

        linkage_ = linkage(M.toarray(), method='ward')
        dendrogram_ = dendrogram(linkage_, no_plot=True)
        clust_index = dendrogram_['leaves']
        M_clust = M[:,clust_index][clust_index,:].toarray()

        test_sparse = to_sparse(pred_avg, pred_coord, shape)
        test_std_sparse = to_sparse(pred_std, pred_coord, shape)
        train_sparse = to_sparse(train_avg, train_coord, shape)
        train_std_sparse = to_sparse(train_std, train_coord, shape)

        pred_clust = (test_sparse + train_sparse)[:,clust_index][clust_index,:].toarray()
        std_clust = (test_std_sparse + train_std_sparse)[:,clust_index][clust_index,:].toarray()

        fig, ax = plt.subplots(3, 3, figsize=(20, 20))

        sns.heatmap(M_clust,robust=True,ax=ax[0,0], square=True,
                    yticklabels=False, xticklabels=False)
        ax[0, 0].set_title('True Matrix')

        sns.heatmap(pred_clust,robust=True,ax=ax[0,1], square=True,
                    yticklabels=False, xticklabels=False)
        ax[0, 1].set_title('Predicted Matrix')

        sns.heatmap(std_clust,robust=True,ax=ax[0,2], cmap='viridis',
                    yticklabels=False, xticklabels=False, square=True)
        ax[0, 2].set_title('Uncertainty (Stdev.)')

        ax[1, 0].scatter(X_test.data, pred_avg, edgecolors=(0, 0, 0))
        ax[1, 0].plot([X_test.data.min(), X_test.data.max()], [pred_avg.min(), pred_avg.max()], 'k--',
                      lw=4)
        ax[1, 0].set_xlabel('Measured')
        ax[1, 0].set_ylabel('Predicted')
        ax[1, 0].set_title('Measured vs Avg. Prediction')

        ax[1, 1].scatter(pred_std, pred_avg, edgecolors=(0, 0, 0))
        ax[1, 1].set_xlabel('Standard Deviation')
        ax[1, 1].set_ylabel('Predicted')
        ax[1, 1].set_title('Stdev. vs Prediction')

        x_ax = np.arange(len(X_test.data))
        sort_vals = np.argsort(X_test.data)
        ax[1, 2].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[1, 2].plot(x_ax, pred_avg[sort_vals], 'rx', alpha=0.5, label='predicted')
        ax[1, 2].set_title('Sorted and overlayed measured and predicted values')
        ax[1, 2].legend()

        ax[2, 0].plot(x_ax, X_test.data[sort_vals], linewidth=4, label="measured")
        ax[2, 0].plot(x_ax, pred_std[sort_vals], 'r', label='stdev.')
        ax[2, 0].set_title('Sorted and overlayed measured stdev values')
        ax[2, 0].legend()

        ax[2, 1].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 1].fill_between(x_ax, pred_avg[sort_vals] - pred_std[sort_vals],
                              pred_avg[sort_vals] + pred_std[sort_vals],
                              alpha=0.5, label="std")
        ax[2, 1].set_title('predicted stdev. relative to predicted value')
        ax[2, 1].legend()

        ax[2, 2].plot(x_ax, X_test.data[sort_vals], label="actual")
        ax[2, 2].fill_between(x_ax, X_test.data[sort_vals] - pred_std[sort_vals],
                              X_test.data[sort_vals] + pred_std[sort_vals],
                              alpha=0.5, label='std')
        ax[2, 2].set_title('predicted stdev. relative to measured value')
        ax[2, 2].legend()

        fig.tight_layout()
        fig.suptitle('{} - NSAMPLES: {} NUM_LATENT: {} SIDE_NOISE: {} NUM_TRAIN {} BURNIN: {} TEST_RATIO {:.5f}\n Metrics - Corr: {:.5f} - Test RMSE: {:.5f}'. \
                     format(self.trainSession.getSaveName().split('.')[0],
                            self.num_samples, self.num_latent, self.side_noise,
                            n_train_examples, self.burnin, test_ratio,
                            test_corr, rmse_avg))

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
            priors=['normal', 'normal'],
            num_latent=self.num_latent,
            burnin=self.burnin,
            nsamples=self.num_samples,
            verbose=self.verbose,
            checkpoint_freq=self.checkpoint_freq,
            save_freq=self.save_freq,
            num_threads=self.num_threads,
            save_name=self.save_name)

        return self
