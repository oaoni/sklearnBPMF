import smurff
from sklearnBPMF.core import Wrapper

class Macau(Wrapper):
    def __init__(self, num_latent=20, burnin=40,
                     num_samples=500, verbose=0, checkpoint_freq=int(10e5),
                     save_freq = -1, save_name="macau_trainable",
                     side_noise=5, side_direct = True, num_threads=1,
                     report_freq=None, metric_mode=1, col_side=True, keep_file=True,tol=1e-9,
                     k_metrics=False, k=20):

        #Macau Hyperparameters
        self.side_noise = side_noise
        self.side_direct = side_direct

        """Constructor""" #Initialize wrapper
        Wrapper.__init__(self,num_latent=num_latent, burnin=burnin,
                         num_samples=num_samples, verbose=verbose, checkpoint_freq=checkpoint_freq,
                         save_freq=save_freq, save_name=save_name, num_threads=num_threads,
                         report_freq=report_freq,metric_mode=metric_mode,col_side=col_side,
                         keep_file=keep_file,tol=tol,
                         k_metrics=k_metrics,k=k)

    def _makeModel(self, train_data, test_data, X_side):

        self.train_coords = [train_data.row, train_data.col]
        self.test_coords = [test_data.row, test_data.col]

        #Add Traiing and test data to training session
        self.trainSession.addTrainAndTest(train_data, test_data,
                                          smurff.AdaptiveNoise(1.0, 10.))
        #Add side information to the training session
        self.trainSession.addSideInfo(0, X_side, noise=smurff.FixedNoise(self.side_noise),
                                      direct = self.side_direct)
        if self.col_side:
            self.trainSession.addSideInfo(1, X_side, noise=smurff.FixedNoise(self.side_noise),
                                          direct = self.side_direct)

        return self

class BPMF(Wrapper):
    def __init__(self, num_latent=20, burnin=40,
                     num_samples=500, verbose=0, checkpoint_freq=int(10e5),
                     save_freq = -1, save_name="bpmf_trainable",
                     side_noise='N/A', side_direct = None, num_threads=1,
                     report_freq=None, metric_mode=1, col_side=None, keep_file=True,tol=1e-9,
                     k_metrics=False, k=20):

        self.side_noise = side_noise
        self.side_direct = side_direct

        """Constructor""" #Initialize wrapper
        Wrapper.__init__(self,num_latent=num_latent, burnin=burnin,
                         num_samples=num_samples, verbose=verbose, checkpoint_freq=checkpoint_freq,
                         save_freq=save_freq, save_name=save_name, num_threads=num_threads,
                         report_freq=report_freq,metric_mode=metric_mode,col_side=col_side,
                         keep_file=keep_file,tol=tol,
                         k_metrics=k_metrics,k=k)

    def _makeModel(self, train_data, test_data, X_side):

        self.train_coords = [train_data.row, train_data.col]
        self.test_coords = [test_data.row, test_data.col]

        #Add Traiing and test data to training session
        self.trainSession.addTrainAndTest(train_data, test_data,
                                          smurff.AdaptiveNoise(1.0, 10.))

        return self
