import torch
import torch.nn as nn


def weighted_CrossEntropyLoss(output, target, classes_weights):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).cuda())
    return cr(output, target)
