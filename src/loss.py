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
from math import exp
from scipy.ndimage.filters import maximum_filter, minimum_filter
import sys

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


class TopologyWarpingBCE(nn.Module):
    """
    Weights topological errors by warping the segmentation onto the ground-truth
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.bce = nn.BCEWithLogitsLoss()
        self.iteration = 1

    def forward(self, pred, label, mask=None):
        sig = 1/(1+exp(-self.iteration/5000+2.3))
        weight = self.weight_topological_errors(pred, label,
                                                weight=sig*10)
        err = - Variable(weight) * (label * -(1 + (-pred).exp()).log() +
                                      (1 - label) * -(1 + pred.exp()).log())

        cost = err.sum()

        self.iteration += 1

        return cost

    def weight_topological_errors(self, pred, label, weight=5):
        if self.iteration < 600:
            return torch.ones(pred.size()).cuda(0)

        iterations = int(max(3, 7*(1-1/(1+exp(-self.iteration/5000+3)))))

        _pred = pred.cpu()
        _pred = _pred.data.numpy() > 0.5
        _label = label.cpu()
        _label = _label.data.numpy() > 0.5

        topological_errors = np.zeros(_pred.shape)

        for i in range(_pred.shape[0]):
            warping = self.warp(_pred[i][0], _label[i][0],
                                iterations=iterations)
            # topological_errors[i][0] = np.bitwise_and(np.bitwise_not(warping),
            #                                           _label[i][0])
            topological_errors[i][0] = np.bitwise_xor(warping, _label[i][0])

        topological_errors = torch.from_numpy(topological_errors.astype(np.float32)).cuda(0)

        return (weight-1)*topological_errors + torch.ones(pred.size()).cuda(0)

    def warp(self, prediction, gt, iterations=1):
        # Make sure both the prediction and ground-truth are binary
        _gt = np.pad(gt > 0.5, pad_width=((1,)), mode='constant').copy().astype(np.uint8)
        _prediction = np.pad(prediction > 0.5,
                            pad_width=((1,)),
                            mode='constant').copy().astype(np.uint8)

        # Get dilated points by max filtering the segmentation
        dilation = maximum_filter(_prediction, size=(3, 3, 3))
        result = np.bitwise_and(dilation, np.bitwise_not(_prediction))

        # Warp simple dilated points onto the ground-truth
        voxel_list = np.nonzero(np.bitwise_and(result, _gt))
        warping = _prediction.copy()

        for z, y, x in zip(*voxel_list):
            neighborhood = warping[z-1:z+2, y-1:y+2, x-1:x+2].copy()
            neighborhood[1, 1, 1] = 1
            if self._is_simple_point(neighborhood):
                warping[z, y, x] = 1

        # Get eroded points by min filtering the segmentation
        erosion = minimum_filter(_prediction, size=(3, 3, 3))
        result = np.bitwise_and(_prediction, np.bitwise_not(erosion))

        # Warp simple eroded points onto the ground-truth
        voxel_list = np.nonzero(np.bitwise_and(result, np.bitwise_not(_gt)))

        for z, y, x in zip(*voxel_list):
            neighborhood = _prediction[z-1:z+2, y-1:y+2, x-1:x+2].copy()
            threshold = neighborhood[1, 1, 1]
            if self._is_simple_point(neighborhood):
                warping[z, y, x] = 0

        if iterations > 1:
            iterations += -1
            return self.warp(warping[1:-1, 1:-1, 1:-1] > 0, gt, iterations=iterations)
        else:
            return warping[1:-1, 1:-1, 1:-1] > 0

    def _is_simple_point(self, neighborhood):
        if np.sum(neighborhood) == 0:
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
            return False

        # Calculates the topological number of the component
        result = (result == threshold)
        result[1, 1, 1] = 0
        labeled_array, num_features = label(result,
                                            structure=np.ones((3, 3, 3)))

        if num_features != 1:
            return False

        return True
