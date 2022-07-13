import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    Copied from
    https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup/cosine_annealing_with_warmup.py

    You can see its learning rate graph at here: https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.01, T_warmup=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_warmup < 0 or not isinstance(T_warmup, int):
            raise ValueError("Expected positive integer T_warmup, but got {}".format(T_warmup))
        self.T_0 = T_0                  # Size of the first epoch cycle
        self.T_mult = T_mult            # Size multiplier of LR cycle
        self.base_eta_max = eta_max     # Initial maximum learning rate
        self.eta_max = eta_max          # Maximum learning rate
        self.T_warmup = T_warmup        # Warm-up epoch
        self.T_i = T_0                  # Size of epoch cycle
        self.gamma = gamma              # Multiplier of maximum learning rate
        self.cycle = 0                  # Current cycle
        self.T_cur = last_epoch         # Current Epoch

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_warmup:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_warmup + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr)
                    * (1 + math.cos(math.pi * (self.T_cur - self.T_warmup) / (self.T_i - self.T_warmup))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_warmup) * self.T_mult + self.T_warmup
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
