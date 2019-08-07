import numpy as np
from scipy import stats
from kdeebm import mixture_model
from kdeebm import mcmc
from matplotlib import pyplot as plt
from kde_ebm_paper.data_gen import get_gamma

from kde_ebm_paper.waic_noxtoby import calculate_waic

from scipy.stats.distributions import chi2
def log_likelihood_ratio(ll1, ll2):
    return(ll1-ll2)
def likelihood_ratio(ll1, ll2):
    llmin = min([ll1,ll2])
    llmax = max([ll1,ll2])
    return(2*(llmax-llmin))
def likelihood_ratio_test(log_lh_1, log_lh_2, dof=1):
    like_ratio = likelihood_ratio(log_lh_1,log_lh_2)
    p = chi2.sf(like_ratio, dof) # dof is difference between DoF for each model
    #print 'p: %.30f' % p
    return p

# from waic_noxtoby import waic
# log_like = -100
# waic,lpd,p_waic,elpd_waic,p_loo,elpd_loo = waic(log_like)


def plot_imshow_results(res, shape_range, sep_range, vmax=None, vmin=None, lab=''):
    fig, ax = plt.subplots()
    im = ax.imshow(res, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(shape_range)))
    ax.set_xticklabels(shape_range)
    ax.set_yticks(np.arange(0, len(sep_range)))
    ax.set_yticklabels([str(x) for x in sep_range])
    ax.set_xlabel('shape parameter')
    ax.annotate("increasing Gaussianity",
                xy=(shape_range[-2], sep_range[5]),
                xytext=(shape_range[3], sep_range[5]),
                arrowprops=dict(arrowstyle='-|>'),
                va='center')
    ax.set_ylabel('separation')
    fig.colorbar(im,label=lab)
    return fig, ax


np.random.seed(42)

# def main():
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
kde_bic_results = np.empty((len(sep_range), len(shape_range), n_repeats, len(h_stage_prob)-1))
gmm_bic_results = np.empty((len(sep_range), len(shape_range), n_repeats, len(h_stage_prob)-1))

gmm_failed_fits = 0
kde_failed_fits = 0

for i in range(len(sep_range)):
    for j in range(len(shape_range)):
        for rep_n in range(n_repeats):
            separation = sep_range[i]
            shape = shape_range[j]
            X, y, stages = get_gamma(n_samples, h_stage_prob,
                                     separation, shape, n_biomarkers=3)
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
                kde_bic = kde_mixtures[k].BIC(X=masked_X)
                gmm_bic = gmm_mixtures[k].BIC(X=masked_X)
                kde_like_results[i, j, rep_n, k] = kde_like
                gmm_like_results[i, j, rep_n, k] = gmm_like
                kde_bic_results[i, j, rep_n, k] = kde_bic
                gmm_bic_results[i, j, rep_n, k] = gmm_bic
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
print('GMM corr=%f, KDE corr=%f (MWU: p=%E)' % (gmm_stage_results.mean(),
                                                kde_stage_results.mean(),
                                                p_val))
p_val = stats.mannwhitneyu(gmm_like_results.flatten(),
                           kde_like_results.flatten()).pvalue
print('GMM like=%f, KDE like=%f (MWU: p=%E)' % (gmm_like_results.mean(),
                                                kde_like_results.mean(),
                                                p_val))
#* BIC
p_val = stats.mannwhitneyu(gmm_bic_results.flatten(),
                           kde_bic_results.flatten()).pvalue
print('GMM BIC=%f, KDE BIC=%f (MWU: p=%E)' % (gmm_bic_results.mean(),
                                              kde_bic_results.mean(),
                                              p_val))
#* WAIC
waic_gmm,lpd_gmm,p_waic_gmm,elpd_waic_gmm = calculate_waic(-1*gmm_like_results.flatten())
waic_kde,lpd_kde,p_waic_kde,elpd_waic_kde = calculate_waic(-1*kde_like_results.flatten())
print('GMM like=%f, KDE like=%f | GMM WAIC = %E (eff. # params p_waic=%E); KDE WAIC = %E (eff. # params p_waic=%E)' % (gmm_like_results.mean(),
                                                                    kde_like_results.mean(),
                                                                    waic_gmm,p_waic_gmm,waic_kde,p_waic_kde))
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
like_diff = kde_like_results - gmm_like_results # negative log-like
like_diff *= -1 # log-like
like_diff = like_diff.mean(axis=(-1, -2)) # avg over events (-1)  and repeats (-2)

#* PLOTS
fig, ax = plot_imshow_results(ebm_res_diff, shape_range, sep_range,
                              vmin=-2, vmax=2, lab='$\Delta$(Kendall tau: KDE MM vs GMM)')
ax.set_title("Sequence correlation with ground truth")
fig.show()
fig.savefig('ebm_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                          n_samples))
min_max = np.abs(stage_res_diff).max()
fig, ax = plot_imshow_results(stage_res_diff, shape_range,
                              sep_range, vmin=-min_max, vmax=min_max,
                              lab='$\Delta$(Spearman rho: KDE MM vs GMM)')
ax.set_title("Staging correlation with ground truth")
fig.show()
fig.savefig('stage_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                            n_samples))
min_max = np.abs(like_diff).max()
fig, ax = plot_imshow_results(like_diff, shape_range,
                              sep_range, vmin=-min_max, vmax=min_max,
                              lab="$\Delta$(Likelihood)")
ax.set_title("Likelihood difference: KDE MM vs GMM")
fig.show()
fig.savefig('like_res_miss-%f-n=%i.png' % (missing_vals_percent,
                                           n_samples))


#* Likelihood Ratio test
# like_ratio = kde_like_results - gmm_like_results # negative log-like
# like_diff *= -1 # log-like
# like_diff = like_diff.mean(axis=(-1, -2)) # avg over n=25 repeats and m=3 events
#* Log-likelihoods
L1 = -1*gmm_like_results #.flatten()
L2 = -1*kde_like_results #.flatten()
L1_L2_ratio = log_likelihood_ratio(L1,L2)
p = chi2.sf(L1_L2_ratio, df=1)

p_av = chi2.sf(L1_L2_ratio.mean(axis=-2), df=1)

# p = np.empty(shape=L1_L2_ratio.shape)
# for k in range(len(p)):
#     #* p-value is negative if GMM is more likely than KDEMM
#     p[k] = (-1)**(L1[k]>=L2[k])*likelihood_ratio_test(L1[k],L2[k])


# fig, ax = plot_imshow_results_subplots(like_diff,
#                                        p.mean(axis=(-2,-1)),
#                                        shape_range,
#                                        sep_range,
#                                        vmin=p_av.min(), vmax=p_av.max())
# ax[1].set_title("Likelihood Ratio test, H_0:GMM, H_1:KDEMM")
# fig.show()

# def plot_imshow_results_subplots(res1, res2, shape_range, sep_range, vmax=None, vmin=None):
#     fig, ax = plt.subplots(1,2,sharey=True)
#     im1 = ax[0].imshow(res1, cmap='bwr', vmin=vmin, vmax=vmax)
#     ax[0].set_xticks(np.arange(0, len(shape_range)))
#     ax[0].set_xticklabels(shape_range)
#     ax[0].set_yticks(np.arange(0, len(sep_range)))
#     ax[0].set_yticklabels([str(x) for x in sep_range])
#     ax[0].set_xlabel('shape parameter')
#     ax[0].set_ylabel('separation')
#     fig.colorbar(im1)
#     im2 = ax[1].imshow(res1, cmap='bwr') #, vmin=vmin, vmax=vmax)
#     ax[1].set_xticks(np.arange(0, len(shape_range)))
#     ax[1].set_xticklabels(shape_range)
#     ax[1].set_yticks(np.arange(0, len(sep_range)))
#     ax[1].set_yticklabels([str(x) for x in sep_range])
#     ax[1].set_xlabel('shape parameter')
#     #ax[1].set_ylabel('separation')
#     return fig, ax








############### https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

orig_cmap = matplotlib.cm.bwr_r
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.05, name='shifted')
shrunk_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=0.05, stop=1, name='shrunk')

##################

def plot_imshow_results2(res, shape_range, sep_range, vmax=None, vmin=None, lab=''):
    fig, ax = plt.subplots()
    im = ax.imshow(res, cmap=shifted_cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, len(shape_range)))
    ax.set_xticklabels(shape_range)
    ax.set_yticks(np.arange(0, len(sep_range)))
    ax.set_yticklabels([str(x) for x in sep_range])
    ax.set_xlabel('shape parameter')
    ax.set_ylabel('separation')
    ax.annotate("increasing Gaussianity",
                xy=(shape_range[-2], sep_range[5]),
                xytext=(shape_range[3], sep_range[5]),
                arrowprops=dict(arrowstyle='-|>'),
                va='center')
    fig.colorbar(im,label=lab)
    return fig, ax

fig, ax = plot_imshow_results2(p.mean(axis=(-2,-1)),
                                       shape_range,
                                       sep_range,
                                       vmin=p_av.min(), vmax=p_av.max(), lab='p value')
ax.set_title("Likelihood Ratio test, H$_0$:GMM, H$_1$:KDEMM")
fig.show()
fig.savefig('like_ratio_test-%f-n=%i.png' % (missing_vals_percent, n_samples))


print('GMM mean(LH)=%f, KDE mean(LH)=%f (mean p=%E)' % (np.mean(L1),
                                         np.mean(L2),
                                         np.mean(p)))




# if __name__ == '__main__':
#     np.random.seed(42)
#     main()
