import torch
import numpy as np


def binarize(note_logit, device):
    note_out = torch.sigmoid(note_logit)
    tmp = torch.zeros_like(note_out)
    tmp[note_out >= 0.5] = 1
    return tmp
    # rand = torch.rand(note_out.size()).to(device)
    # prob = note_out - rand
    # note_tmp = torch.zeros_like(prob)
    # note_tmp[prob > 0] = 1

    # return note_tmp


def kl_loss(mu, var):
    return -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())


def note_constrained_loss(recon, orig, note):
    """
    Loss only valid when note == 1
    """
    denominator = note.sum()
    if denominator.item() == 0:
        return denominator
    else:
        return (note * (orig - recon) ** 2).sum() / denominator


def compute_note_density_idx(note, note_density_classes):
    seq_len = note.shape[0]
    n_drum_classes = note.shape[1]
    note_density = np.sum(note) / (seq_len * n_drum_classes)
    if note_density > note_density_classes[-1]:
        note_density_idx = len(note_density_classes) - 1
    else:
        for i, d in enumerate(note_density_classes):
            if d > note_density:
                note_density_idx = i
                break
    return note_density_idx


def compute_vel_contour(vel, note, vel_classes):
    if np.sum(note[:, 2]) > np.sum(note[:, 5]):
        accent = vel[:, 2]
    else:
        accent = vel[:, 5]

    # accent = np.sum(vel, axis=-1) / np.sum(note, axis=-1)
    # accent = np.nan_to_num(accent)

    vel_classes_ = np.expand_dims(vel_classes, 1)

    # round
    accent = np.around(accent, decimals=1)

    accent = np.expand_dims(accent, 0)

    accent = np.repeat(accent, 5, axis=0)

    idx = (accent <= vel_classes_) * 1

    idx_sum = np.sum(idx, axis=0)

    vel_idx = 5 - idx_sum

    vel_contour = np.eye(len(vel_classes_))[vel_idx]
    return vel_contour


def compute_laidbackness(mt, note):

    avg_timing = np.sum(mt, axis=1) / np.sum(note, axis=1)
    avg_timing = np.nan_to_num(avg_timing, 0)
    timing = np.zeros_like(avg_timing)
    timing[avg_timing < 0] = 0
    timing[avg_timing == 0] = 1
    timing[avg_timing > 0] = 2
    timing = timing.astype(int)
    mt_contour = np.identity(3)[timing]
    # print (mt_contour)
    return mt_contour


def kl_divergence(mean1, std1, mean2, std2):

    kld_mv = np.log(std2 / std1) + (std1**2 + (mean1 - mean2)
                                    ** 2) / (2 * std2**2) - 0.5

    return kld_mv


def compute_metrical_mean_std(inputs, onset_inputs, metric_pos):
    """
    inputs : velocity or microtiming matrix (shape=(data_len,e seq_len, n_drum_classes))
    onset_inputs : binary matrix indicating onset location  (shape=(data_len, seq_len, n_drum_classes)). same shape as input
    metric_pos : 1,2,3,4.. (1/16th note position)
    """

    assert inputs.shape == onset_inputs.shape
    assert metric_pos < len(inputs)

    m = metric_pos - 1

    curr_input = inputs[:, m::m + 4].reshape(-1)
    curr_onset = onset_inputs[:, m::m + 4].reshape(-1)

    # only select values where onset = 1
    curr_input_valid = curr_input[curr_onset > 0]
    # print (curr_input_valid)
    # print ("kl", len(curr_input_valid))

    return np.mean(curr_input_valid), np.std(curr_input_valid)


def compute_metrical_kl(pred, orig, note_pred, note_orig):
    # input_shape : (# data, seq_len, n_drum_classes)
    # print ("orig range",np.min(orig), np.max(orig))

    p1_mean, p1_std = compute_metrical_mean_std(pred, note_pred, 1)
    p2_mean, p2_std = compute_metrical_mean_std(pred, note_pred, 2)
    p3_mean, p3_std = compute_metrical_mean_std(pred, note_pred, 3)
    p4_mean, p4_std = compute_metrical_mean_std(pred, note_pred, 4)

    o1_mean, o1_std = compute_metrical_mean_std(orig, note_orig, 1)
    o2_mean, o2_std = compute_metrical_mean_std(orig, note_orig, 2)
    o3_mean, o3_std = compute_metrical_mean_std(orig, note_orig, 3)
    o4_mean, o4_std = compute_metrical_mean_std(orig, note_orig, 4)

    kl1 = kl_divergence(p1_mean, p1_std, o1_mean, o1_std)
    kl2 = kl_divergence(p2_mean, p2_std, o2_mean, o2_std)
    kl3 = kl_divergence(p3_mean, p3_std, o3_mean, o3_std)
    kl4 = kl_divergence(p4_mean, p4_std, o4_mean, o4_std)

    return (kl1 + kl2 + kl3 + kl4) / 4


def tensor2np(t):
    """ Pytorch GPU tensor on graph to numpy """
    return t.detach().cpu().numpy()


def batch2array(inputs):
    """ Reshape list of batched arrays to list of arrays
    inputs: list ; shape = (n_of_batches, batch_size, seq_len, n_drum_classes)
    outputs: numpy array ; shape = (n_of_batches * batch_size, seq_len, n_drum_classes)
    """
    outputs = np.array(inputs)
    outputs = np.vstack(outputs)
    return outputs


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L
