import numpy as np
from scipy import stats
from kdeebm import mixture_model
from kdeebm import mcmc
from matplotlib import pyplot as plt
from data_gen import get_gamma


def plot_imshow_results(res, shape_range, sep_range, vmax=None, vmin=None):
    fig, ax = plt.subplots()
    im = ax.imshow(res, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(shape_range)))
    ax.set_xticklabels(shape_range)
    ax.set_yticks(np.arange(0, len(sep_range)))
    ax.set_yticklabels([str(x) for x in sep_range])
    ax.set_xlabel('shape parameter')
    ax.set_ylabel('seperation')
    fig.colorbar(im)
    return fig, ax


def main():
    sep_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    shape_range = [0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10]
    h_stage_prob = [0.55, 0.25, 0.15, 0.05]
    n_samples = 200
    n_repeats = 25
    missing_vals_percent = 0.1

    kde_ebm_results = np.empty((len(sep_range), len(shape_range), n_repeats))
    gmm_ebm_results = np.empty((len(sep_range), len(shape_range), n_repeats))
    kde_stage_results = np.empty((len(sep_range), len(shape_range), n_repeats))
    gmm_stage_results = np.empty((len(sep_range), len(shape_range), n_repeats))
    kde_like_results = np.empty((len(sep_range), len(shape_range),
                                 n_repeats, len(h_stage_prob)-1))
    gmm_like_results = np.empty((len(sep_range), len(shape_range),
                                 n_repeats, len(h_stage_prob)-1))

    gmm_failed_fits = 0
    kde_failed_fits = 0

    for i in range(len(sep_range)):
        for j in range(len(shape_range)):
            for rep_n in range(n_repeats):
                seperation = sep_range[i]
                shape = shape_range[j]
                X, y, stages = get_gamma(n_samples, h_stage_prob,
                                         seperation, shape, n_biomarkers=3)

                if missing_vals_percent > 0:
                    drop_vals = np.random.randint(low=0, high=X.size-1,
                                                  size=int(missing_vals_percent*X.size))
                    x_idx, y_idx = np.unravel_index(drop_vals, X.shape)
                    X[x_idx, y_idx] = np.nan
                y = y.astype(int)
                gmm_mixtures = mixture_model.fit_all_gmm_models(X, y)
                kde_mixtures = mixture_model.fit_all_kde_models(X, y)
                for k in range(len(gmm_mixtures)):
                    masked_X = X[:, k]
                    masked_X = masked_X[~np.isnan(masked_X)].reshape(-1, 1)
                    kde_like = kde_mixtures[k].likelihood(masked_X)
                    gmm_like = gmm_mixtures[k].likelihood(gmm_mixtures[k].theta, masked_X)
                    kde_like_results[i, j, rep_n, k] = kde_like
                    gmm_like_results[i, j, rep_n, k] = gmm_like
                seq = np.arange(X.shape[1])
                np.random.shuffle(seq)
                g_truth = seq.argsort()
                X = X[:, seq]
                gmm_res = mcmc.enumerate_all(X, gmm_mixtures)
                kde_res = mcmc.enumerate_all(X, kde_mixtures)
                if gmm_res is None and kde_res is None:
                    gmm_failed_fits += 1
                    kde_failed_fits += 1
                    continue
                elif gmm_res is None:
                    gmm_failed_fits += 1
                    continue
                elif kde_res is None:
                    kde_failed_fits += 1
                    continue
                gmm_kt = stats.kendalltau(gmm_res.ordering, g_truth)[0]
                kde_kt = stats.kendalltau(kde_res.ordering, g_truth)[0]
                kde_ebm_results[i, j, rep_n] = kde_kt
                gmm_ebm_results[i, j, rep_n] = gmm_kt

                kde_prob_mat = mixture_model.get_prob_mat(X, kde_mixtures)
                kde_stages, stages_like = kde_res.stage_data(kde_prob_mat)
                gmm_prob_mat = mixture_model.get_prob_mat(X, gmm_mixtures)
                gmm_stages, stages_like = kde_res.stage_data(gmm_prob_mat)
                kde_stage_corr = stats.spearmanr(stages, kde_stages)[0]
                gmm_stage_corr = stats.spearmanr(stages, gmm_stages)[0]
                kde_stage_results[i, j, rep_n] = kde_stage_corr
                gmm_stage_results[i, j, rep_n] = gmm_stage_corr
    p_val = stats.mannwhitneyu(gmm_stage_results.flatten(),
                               kde_stage_results.flatten()).pvalue

    print('GMM corr=%f, KDE corr=%f (p=%E)' % (gmm_stage_results.mean(),
                                               kde_stage_results.mean(),
                                               p_val))
    p_val = stats.mannwhitneyu(gmm_like_results.flatten(),
                               kde_like_results.flatten()).pvalue
    print('GMM like=%f, KDE like=%f (p=%E)' % (gmm_like_results.mean(),
                                               kde_like_results.mean(),
                                               p_val))
    p_val = stats.mannwhitneyu(gmm_ebm_results.flatten(),
                               kde_ebm_results.flatten()).pvalue
    print('GMM tau=%f, KDE tau=%f (p=%E)' % (gmm_ebm_results.mean(),
                                             kde_ebm_results.mean(),
                                             p_val))
    print('GMM failed %i, KDE failed %i' % (gmm_failed_fits, kde_failed_fits))
    ebm_res_diff = (kde_ebm_results.mean(axis=-1) -
                    gmm_ebm_results.mean(axis=-1))
    stage_res_diff = (kde_stage_results.mean(axis=-1) -
                      gmm_stage_results.mean(axis=-1))
    like_diff = kde_like_results - gmm_like_results
    like_diff *= -1
    like_diff = like_diff.mean(axis=(-1, -2))
    fig, ax = plot_imshow_results(ebm_res_diff, shape_range, sep_range,
                                  vmin=-2, vmax=2)
    fig.savefig('ebm_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                              n_samples))
    min_max = np.abs(stage_res_diff).max()
    fig, ax = plot_imshow_results(stage_res_diff, shape_range,
                                  sep_range, vmin=-min_max, vmax=min_max)
    fig.savefig('stage_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                                n_samples))
    min_max = np.abs(like_diff).max()
    fig, ax = plot_imshow_results(like_diff, shape_range,
                                  sep_range, vmin=-min_max, vmax=min_max)
    fig.savefig('like_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                               n_samples))
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    main()
