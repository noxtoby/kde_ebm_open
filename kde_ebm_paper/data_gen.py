from scipy import stats
import numpy as np


def stages_to_data(stages, seperation, shape, n_biomarkers=3):
    bm_data = np.zeros((stages.shape[0], n_biomarkers))
    for i in range(stages.shape[0]):
        bm_data[i, :stages[i]] = 1
    n_disease = bm_data.sum().astype(int)
    h_values = np.random.gamma(shape, size=bm_data.size-n_disease)
    d_values = np.random.gamma(shape, size=n_disease)
    mean = stats.gamma.mean(shape)
    std = stats.gamma.std(shape)
    h_values -= mean
    d_values -= mean
    d_values *= -1
    d_values += 2*seperation*std
    bm_data[bm_data == 0] = h_values
    bm_data[bm_data == 1] = d_values
    return bm_data


def get_gamma(n_samples, stage_probs, seperation, shape, n_biomarkers=3):
    h_stage_prob = stage_probs
    d_stage_prob = stage_probs[::-1]
    h_stage_cumprob = np.cumsum(h_stage_prob)
    d_stage_cumprob = np.cumsum(d_stage_prob)

    stage_probs = np.random.random(n_samples)
    h_stages = stage_probs[:n_samples//2, np.newaxis]
    rows, columns = np.where(h_stages < h_stage_cumprob)
    h_stages = h_stages.flatten()
    for i in range(h_stages.shape[0]):
        h_stages[i] = np.min(columns[rows == i])
    h_stages = h_stages.astype(int)
    # Repeated code, bad
    d_stages = stage_probs[n_samples//2:, np.newaxis]
    rows, columns = np.where(d_stages < d_stage_cumprob)
    d_stages = d_stages.flatten()
    for i in range(d_stages.shape[0]):
        d_stages[i] = np.min(columns[rows == i])
    d_stages = d_stages.astype(int)

    all_stages = np.hstack((h_stages, d_stages))
    y = np.hstack((np.zeros(n_samples//2),
                   np.ones(n_samples//2)))
    X = stages_to_data(all_stages, seperation, shape)
    return X, y, all_stages
