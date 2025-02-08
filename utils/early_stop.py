import numpy as np
import torch
import os
class EarlyStopping:
    """Early stops the training if monitor value doesn't improve after a given patience."""
    def __init__(self, minmax = 'max',patience=10, delta=1e-6, path='results/experiments', verbose=True,trace_func=print, save_every_eposh=False, model_name=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            minmax (str): 'max' or 'min'
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
                       
        """
        self.patience = patience
        self.minmax = minmax
        self.counter = 0
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.check_time = 0
        self.save_name = None
        self.save_name_epoch = None
        self.verbose = verbose
        self.last_save_name = None
        self.last_save_name_epoch = None
        self.last_save_time = 0
        self.save_every_eposh = save_every_eposh
        self.model_name = model_name
    def __call__(self, monitor, model):
        self.check_time += 1
        check_monitor = monitor
        if self.best_score is None:
            self.best_score = check_monitor
            self.save_checkpoint(monitor, model)
        elif self.minmax == 'max' and check_monitor < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.minmax == 'min' and check_monitor > self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = check_monitor
            self.save_checkpoint(monitor, model)
            self.counter = 0

        if self.save_every_eposh == True:
            self.save_checkpoint_epoch(monitor, model)

        return self.early_stop
    
    def save_checkpoint_epoch(self, monitor, model):
        '''Saves model every epoch'''
        self.save_name_epoch = os.path.join(self.path,'{}_checkpoint_epoch_{}.pt'.format(self.model_name, self.check_time))
        if self.verbose:
            self.trace_func(f'Monitor ({monitor:.6f}), epoch: {self.check_time}.  Saving model {self.save_name_epoch} ...')
        
        torch.save(model.state_dict(), self.save_name_epoch)
        if self.last_save_name_epoch is not None:
            os.remove(self.last_save_name_epoch)
        self.last_save_name_epoch = self.save_name_epoch

    def save_checkpoint(self, monitor, model):
        '''Saves model when validation loss decrease.'''
        self.last_save_time = self.check_time
        if self.model_name is not None:
            self.save_name = os.path.join(self.path,'{}_best_checkpoint_epoch_{}.pt'.format(self.model_name, self.check_time))
        else:
            self.save_name = os.path.join(self.path,'best_checkpoint_epoch_{}.pt'.format(self.check_time))
            
        if self.verbose:
            self.trace_func(f'Monitor reachs optimal ({monitor:.6f}), epoch: {self.check_time}.  Saving model {self.save_name} ...')
        
        torch.save(model.state_dict(), self.save_name)
        if self.last_save_name is not None:
            os.remove(self.last_save_name)
        self.last_save_name = self.save_name

    def load_best_checkpoint(self,model):
        self.trace_func(f'Loading Best Checkpoint {self.save_name} ...')
        model.load_state_dict(torch.load(self.save_name))
        return model
