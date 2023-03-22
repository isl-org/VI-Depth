import numpy as np
import torch

def rmse(estimate, target):
    return np.sqrt(np.mean((estimate - target) ** 2))

def mae(estimate, target):
    return np.mean(np.abs(estimate - target))

def absrel(estimate, target):
    return np.mean(np.abs(estimate - target) / target)

def inv_rmse(estimate, target):
    return np.sqrt(np.mean((1.0/estimate - 1.0/target) ** 2))

def inv_mae(estimate, target):
    return np.mean(np.abs(1.0/estimate - 1.0/target))

def inv_absrel(estimate, target):
    return np.mean((np.abs(1.0/estimate - 1.0/target)) / (1.0/target))

class ErrorMetrics(object):
    def __init__(self):
        # initialize by setting to worst values
        self.rmse, self.mae, self.absrel = np.inf, np.inf, np.inf
        self.inv_rmse, self.inv_mae, self.inv_absrel = np.inf, np.inf, np.inf

    def compute(self, estimate, target, valid):
        # apply valid masks
        estimate = estimate[valid]
        target = target[valid]

        # estimate and target will be in inverse space, convert to regular
        estimate = 1.0/estimate
        target = 1.0/target

        # depth error, estimate in meters, convert units to mm
        self.rmse = rmse(1000.0*estimate, 1000.0*target)
        self.mae = mae(1000.0*estimate, 1000.0*target)
        self.absrel = absrel(1000.0*estimate, 1000.0*target)

        # inverse depth error, estimate in meters, convert units to 1/km
        self.inv_rmse = inv_rmse(0.001*estimate, 0.001*target)
        self.inv_mae = inv_mae(0.001*estimate, 0.001*target)
        self.inv_absrel = inv_absrel(0.001*estimate, 0.001*target)

class ErrorMetricsAverager(object):
    def __init__(self):
        # initialize avg accumulators to zero
        self.rmse_avg, self.mae_avg, self.absrel_avg = 0, 0, 0
        self.inv_rmse_avg, self.inv_mae_avg, self.inv_absrel_avg = 0, 0, 0
        self.total_count = 0

    def accumulate(self, error_metrics):
        # adds to accumulators from ErrorMetrics object
        assert isinstance(error_metrics, ErrorMetrics)

        self.rmse_avg += error_metrics.rmse
        self.mae_avg += error_metrics.mae
        self.absrel_avg += error_metrics.absrel

        self.inv_rmse_avg += error_metrics.inv_rmse
        self.inv_mae_avg += error_metrics.inv_mae
        self.inv_absrel_avg += error_metrics.inv_absrel

        self.total_count += 1

    def average(self):
        # print(f"Averaging depth metrics over {self.total_count} samples")
        self.rmse_avg = self.rmse_avg / self.total_count
        self.mae_avg = self.mae_avg / self.total_count
        self.absrel_avg = self.absrel_avg / self.total_count
        # print(f"Averaging inv depth metrics over {self.total_count} samples")
        self.inv_rmse_avg = self.inv_rmse_avg / self.total_count
        self.inv_mae_avg = self.inv_mae_avg / self.total_count
        self.inv_absrel_avg = self.inv_absrel_avg / self.total_count