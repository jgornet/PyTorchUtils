#!/usr/bin/env python2
__doc__ = """

Loss functions.

Nicholas Turner <nturner.cs@princeton.edu>, 2017
"""

import torch
from torch import nn
import numpy as np
from scipy.ndimage.measurements import label
from torch.autograd import Variable
import torch.nn.functional as F

class BinomialCrossEntropyWithLogits(nn.Module):
    """ 
    A version of BCE w/ logits with the ability to mask
    out regions of output
    """

    def __init__(self):

      nn.Module.__init__(self)

    def forward(self, pred, label, mask=None):

      #Need masking for this application
      # copied from PyTorch's github repo
      # neg_abs = - pred.abs()
      # err = pred.clamp(min=0) - pred * label + (1 + neg_abs.exp()).log()

      # if mask is None:
      #   cost = err.sum() #/ np.prod(err.size())
      # else:
      #   cost = (err * mask).sum() #/ mask.sum()
        err = - (label * -(1 + (-pred).exp()).log() +
                (1 - label) * -(1 + pred.exp()).log())

        cost = err.sum()

        return cost


class WeightedBinomialCrossEntropyWithLogits(nn.Module):
    """
    A version of BCE w/ logits that weights the regions of output
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, pred, label, mask=None):
        neg_abs = - pred.abs()

        if mask is None:
            err = pred.clamp(min=0) - pred * label + (1 + neg_abs.exp()).log()
            cost = err.sum()
        else:
            weighted_ones = label.mean() * (label.new_ones(label.size()) - label)
            weighted_zeros = label.mean() * label
            err = - (weighted_ones * label * pred.log()) \
                  - (weighted_zeros * (label.new_ones(label.size()) - label) * (1 - pred).log())
            cost = err.sum()

        return cost


class SimpleBinomialCrossEntropyWithLogits(nn.Module):
    """
    Weights the simple points in a thresheld input
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, pred, label, mask=None):
        # non_simple_pred = self.label_nonsimple_points(pred)
        # non_simple_label = self.label_nonsimple_points(label)
        # non_simple_points = non_simple_label + non_simple_pred
        # simple_points = torch.ones(label.size()).cuda(0) - non_simple_points
        # weights = 10*non_simple_points + 1*simple_points

        weighted_pred = self.weight_tensor(pred)
        weighted_label = self.weight_tensor(label)

        # neg_abs = - weighted_pred.abs()
        # err = weighted_pred.clamp(min=0) - weighted_pred * weighted_label + \
        #                                (1 + neg_abs.exp()).log()

        # cost = err.sum()
        # cost = nn.BCEWithLogitsLoss.forward(self, weighted_pred, weighted_label)

        # err = - Variable(weights)*(label * -(1 + (-pred).exp()).log() +
        #                  (1 - label) * -(1 + pred.exp()).log())

        err = - (weighted_label * -(1 + (-weighted_pred).exp()).log() +
                 (1 - weighted_label) * -(1 + weighted_pred.exp()).log())

        cost = err.sum()

        return cost
              

    def weight_tensor(self, inputs, simple_weight=1, non_simple_weight=10):
        non_simple_points = self.label_nonsimple_points(inputs)
        simple_points = torch.ones(inputs.size()).cuda(0) - non_simple_points
        inputs_weights = simple_weight * simple_points + \
                         non_simple_weight * non_simple_points
        result = Variable(inputs_weights) * inputs
        return result

    def label_nonsimple_points(self, tensor, threshold=0.5):
        array = tensor.cpu()
        array = array.data.numpy()
        array = (array > threshold)
        labeled_array, num_features = label(array)
        size = labeled_array.shape
        padded_array = np.pad(labeled_array, (1,), 'edge')
        result = np.zeros(size)

        for k in range(0, size[0]):
            for j in range(0, size[1]):
                for i in range(0, size[2]):
                    if self._is_nonsimple_point(padded_array[k:k+3,
                                                             j:j+3,
                                                             i:i+3]):
                        result[k, j, i] = 1

        result = torch.from_numpy(result.astype(np.float32)).cuda(0)

        return result

    def _is_nonsimple_point(self, neighborhood):
        # Skip if the point is background
        if (neighborhood[1, 1, 1] == 0).any():
            return False

        # Setup neighborhood
        result = np.copy(neighborhood)
        threshold = result[1, 1, 1]

        # Create 18-neighborhood structure
        s = np.zeros((3, 3, 3))
        s[0, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        s[1, :, :] = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])
        s[2, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

        # Calculates the topological number of the cavity
        result[result == 0] = -1
        labeled_array, num_features = label(result != threshold,
                                            structure=s)

        if num_features != 1:
            return True

        # Calculates the topological number of the component
        result = (result == threshold)
        result[1, 1, 1] = 0
        labeled_array, num_features = label(result,
                                            structure=np.ones((3, 3, 3)))

        if num_features != 1:
            return True

        return False


class SimpleMSE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, pred, label, mask=None):
        weighted_pred = self.weight_tensor(pred)
        weighted_label = self.weight_tensor(label)

        err = (weighted_pred - weighted_label)
        err = err.pow(2)
        err = err.mean()

        return err

    def weight_tensor(self, inputs, simple_weight=1, non_simple_weight=10):
        non_simple_points = self.label_nonsimple_points(inputs)
        simple_points = torch.ones(inputs.size()).cuda(0) - non_simple_points
        inputs_weights = simple_weight * simple_points + \
                         non_simple_weight * non_simple_points
        result = Variable(inputs_weights) * inputs
        return result

    def label_nonsimple_points(self, tensor, threshold=0.5):
        array = tensor.cpu()
        array = array.data.numpy()
        array = (array > threshold)
        labeled_array, num_features = label(array)
        size = labeled_array.shape
        padded_array = np.pad(labeled_array, (1,), 'edge')
        result = np.zeros(size)

        for k in range(0, size[0]):
            for j in range(0, size[1]):
                for i in range(0, size[2]):
                    if self._is_nonsimple_point(padded_array[k:k+3,
                                                             j:j+3,
                                                             i:i+3]):
                        result[k, j, i] = 1

        result = torch.from_numpy(result.astype(np.float32)).cuda(0)

        return result

    def _is_nonsimple_point(self, neighborhood):
        # Skip if the point is background
        if (neighborhood[1, 1, 1] == 0).any():
            return False

        # Setup neighborhood
        result = np.copy(neighborhood)
        threshold = result[1, 1, 1]

        # Create 18-neighborhood structure
        s = np.zeros((3, 3, 3))
        s[0, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        s[1, :, :] = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])
        s[2, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

        # Calculates the topological number of the cavity
        result[result == 0] = -1
        labeled_array, num_features = label(result != threshold,
                                            structure=s)

        if num_features != 1:
            return True

        # Calculates the topological number of the component
        result = (result == threshold)
        result[1, 1, 1] = 0
        labeled_array, num_features = label(result,
                                            structure=np.ones((3, 3, 3)))

        if num_features != 1:
            return True

        return False
