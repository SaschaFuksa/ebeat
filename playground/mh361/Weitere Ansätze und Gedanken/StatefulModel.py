import numpy as np
import sys
import matplotlib.pyplot as plt

class StatefulModel(object):
    def __init__(self, model, print_val_every=500):
        '''
        model must be stateful keras model object
        batch_input_shape must be specified
        '''
        bis = model.layers[0].get_config()["batch_input_shape"]
        print("batch_input_shape={}".format(bis))
        self.batch_size = bis[0]
        self.ts = bis[1]
        self.Nfeat = bis[2]
        self.model = model
        self.print_val_every = print_val_every

    def get_mse(self, true, est, w=None):
        '''
        calculate MSE for weights == 1
        '''
        if w is None:
            w = np.zeros(true.shape)
            w[:] = 1
        ytrue = true[w == 1].flatten()
        yest = est[w == 1].flatten()

        SSE = np.sum((ytrue - yest) ** 2)
        N = np.sum(w == 1)
        MSE = SSE / N
        return MSE, (SSE, N)

    def X_val_shape_adj(self, X, X_val_orig, y_val_orig, w_val_orig):
        '''
        Make the dimension of X_val the same as the dimension of X
        by adding zeros.

        It is assumed that:
        X_val.shape[i] < X.shape[i]
        i = 0, 1, 2
        '''
        X_val = np.zeros(X.shape)
        X_val[:X_val_orig.shape[0]] = X_val_orig

        myshape = list(y_val_orig.shape)
        myshape[0] = X.shape[0]
        y_val = np.zeros(myshape)
        y_val[:y_val_orig.shape[0]] = y_val_orig

        myshape = list(w_val_orig.shape)
        myshape[0] = X.shape[0]
        w_val = np.zeros(myshape)
        w_val[:w_val_orig.shape[0]] = w_val_orig

        return X_val, y_val, w_val

    def train1epoch(self, X, y, w, epoch=None):
        '''
        devide the training set of time series into batches.
        '''
        print
        "  Training.."
        batch_index = np.arange(X.shape[0])
        ## shuffle to create batch containing different time series
        np.random.shuffle(batch_index)
        count = 1
        for ibatch in range(self.batch_size,
                            X.shape[0] + 1,
                            self.batch_size):

            print
            "    Batch {:02d}".format(count)
            pick = batch_index[(ibatch - self.batch_size):ibatch]
            if len(pick) < self.batch_size:
                continue
            X_batch = X[pick]
            y_batch = y[pick]
            w_batch = w[pick]
            self.fit_across_time(X_batch, y_batch, w_batch, epoch, ibatch)
            count += 1

    def fit_across_time(self, X, y, w, epoch=None, ibatch=None):
        '''
        training for the given set of time series
        It always starts at the time point 0 so we need to reset states to zero.
        '''
        self.model.reset_states()
        for itime in range(self.ts, X.shape[1] + 1, self.ts):
            ## extract sub time series
            Xtime = X[:, itime - self.ts:itime, :]
            ytime = y[:, itime - self.ts:itime, :]
            wtime = w[:, itime - self.ts:itime]
            if np.all(wtime == 0):
                continue
            val = self.model.fit(Xtime, ytime,
                                 nb_epoch=1,
                                 ## no shuffling across rows (i.e. time series)
                                 shuffle=False,
                                 ## use all the samples in one epoch
                                 batch_size=X.shape[0],
                                 sample_weight=wtime,
                                 verbose=False)
            if itime % self.print_val_every == 0:
                print
                "      {start:4d}:{end:4d} loss={val:.3f}".format(
                    start=itime - self.ts, end=itime, val=val.history["loss"][0])
                sys.stdout.flush()
                ## uncomment below if you do not want to save weights for every epoch every batch and every time
                if epoch is not None:
                    self.model.save_weights(
                        "weights_epoch{:03d}_batch{:01d}_time{:04d}.hdf5".format(epoch, ibatch, itime))

    def validate1epoch(self, X_val_adj, y_val_adj, w_val_adj):
        batch_index = np.arange(X_val_adj.shape[0])
        print
        " Validating.."
        val_loss = 0
        count = 1
        for ibatch in range(self.batch_size,
                            X_val_adj.shape[0] + 1,
                            self.batch_size):

            pick = batch_index[(ibatch - self.batch_size):ibatch]
            if len(pick) < self.batch_size:
                continue
            X_val_adj_batch = X_val_adj[pick]
            y_val_adj_batch = y_val_adj[pick]
            w_val_adj_batch = w_val_adj[pick]
            if np.all(w_val_adj_batch == 0):
                continue
            print
            "    Batch {}".format(count)
            SSE, N = self.validate_across_time(
                X_val_adj_batch,
                y_val_adj_batch,
                w_val_adj_batch
            )
            val_loss += SSE
            count += N
        val_loss /= count

        return val_loss

    def validate_across_time(self, X_val_adj, y_val_adj, w_val_adj):

        y_pred_adj = np.zeros(y_val_adj.shape)
        y_pred_adj[:] = np.NaN
        self.model.reset_states()
        for itime in range(self.ts,
                           X_val_adj.shape[1] + 1,
                           self.ts):
            y_pred_adj[:, itime - self.ts:itime, :] = self.model.predict(
                X_val_adj[:, itime - self.ts:itime, :],
                batch_size=X_val_adj.shape[0])

            loss, _ = self.get_mse(y_pred_adj[:, itime - self.ts:itime, :],
                                   y_val_adj[:, itime - self.ts:itime, :],
                                   w_val_adj[:, itime - self.ts:itime])
            if itime % self.print_val_every == 0:
                print
                "     {start:4d}:{end:4d} val_loss={a:.3f}".format(
                    start=itime - self.ts, end=itime, a=loss)
                sys.stdout.flush()

        ## comment out the lines below if you do not want to see the trajectory plots
        fig = plt.figure(figsize=(12, 2))
        nplot = 3
        for i in range(nplot):
            ax = fig.add_subplot(1, nplot, i + 1)
            ax.plot(y_pred_adj[i, :, 0], label="ypred")
            ax.plot(y_val_adj[i, :, 0], label="yval")
            ax.set_ylim(-0.5, 0.5)
        plt.legend()
        plt.show()

        _, (SSE, N) = self.get_mse(y_pred_adj[:, :itime, :],
                                   y_val_adj[:, :itime, :],
                                   w_val_adj[:, :itime])
        return SSE, N

    def fit(self,
            X, y, w, X_val, y_val, w_val,
            Nepoch=300):

        X_val_adj, y_val_adj, w_val_adj = self.X_val_shape_adj(
            X, X_val, y_val, w_val)

        past_val_loss = np.Inf
        history = []
        for iepoch in range(Nepoch):
            self.model.reset_states()
            print
            "__________________________________"
            print
            "Epoch {}".format(iepoch + 1)

            self.train1epoch(X, y, w, iepoch)

            val_loss = self.validate1epoch(X_val_adj,
                                           y_val_adj,
                                           w_val_adj)
            print
            "-----------------> Epoch {iepoch:d} overall valoss={loss:.6f}".format(
                iepoch=iepoch + 1, loss=val_loss),

            ## uncomments here if you want to save weights only when the weights resulted in lower validation loss
            ##if val_loss < past_val_loss:
            ##    print ">>SAVED<<"
            ##    self.model.save_weights("/model.h5")
            ##    past_val_loss = val_loss
            ##
            print
            ""
            history.append(val_loss)

        return history