import torch


def diceutil(result, reference):
    smooth = 0.001
    intersection = result*reference
    intersection_sum = torch.sum(intersection)
    reference_sum = reference.sum()
    result_sum = result.sum()
    dice = 2*intersection_sum / (result_sum + reference_sum +smooth)

    return dice


def precisionutil(result, reference):
    smooth = 0.001
    intersection = result * reference
    intersection_sum = torch.sum(intersection)
    reference_sum = reference.sum()
    result_sum = result.sum()
    precision = intersection_sum / (result_sum + smooth)

    return precision



def recallutil(result, reference):
    smooth = 0.001
    intersection = result * reference
    intersection_sum = torch.sum(intersection)
    reference_sum = reference.sum()
    result_sum = result.sum()
    recall = intersection_sum / (reference_sum + smooth)

    return  recall