import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# #*** Exemplar data
# fig,ax = plt.subplots(2,2)
# shapes = [1.5,10]
# scales = [2,1]
# for kappa,i in zip(shapes,range(len(shapes))):
#     for scale,j in zip(scales,range(len(scales))):
#         x_hc = np.random.gamma(shape=kappa,scale=scale,size=(5000,1))
#         x_ad = np.random.gamma(shape=kappa,scale=scale,size=(5000,1))
#         sigma = np.std(x_hc)
#         x_hc = x_hc - np.mean(x_hc)
#         x_ad = x_ad - np.mean(x_ad)
#         f = sigma*(2-i)*(2-j)
#         x_ad = -x_ad + f
#         ax[i,j].hist(np.concatenate((x_hc,x_ad),axis=1),label=['HC','AD'],bins=24 - (j%2)*8)
#         ax[i,j].legend()
#         ax[i,j].set_title('Shape = {0}, Scale = {1}, f = {2}'.format(kappa,scale,np.around(f,2)))
# fig.show()
# # fig.savefig('fig1_exemplardata.png')

#*** Exemplar data v2
fig,ax = plt.subplots(2,2,figsize=(10,4))
shapes = [1.5,10]
scale = 1.0
sf = [5,1]
for kappa,i in zip(shapes,range(len(shapes))):
    for f,j in zip(sf,range(len(sf))):
        x_hc = np.random.gamma(shape=kappa,scale=scale,size=(5000,1))
        x_ad = np.random.gamma(shape=kappa,scale=scale,size=(5000,1))
        x_hc = x_hc - np.mean(x_hc)
        x_ad = x_ad - np.mean(x_ad)
        x_ad = -x_ad + f
        x = np.concatenate((x_hc,x_ad),axis=1)
        ax[i,j].hist(x,label=['HC','AD'],bins=24 - 0*(j%2)*4)
        if (i==1) & (j==1):
            ax[i,j].legend(fontsize=12)
        #ax[i,j].set_title('Shape = {0}, Scale = {1}, f = {2}'.format(kappa,scale,np.around(f,2)))
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim(left=np.min(x.flatten()), right=np.max(x.flatten()))
        
        fig2,ax2 = plt.subplots(figsize=(10,4))
        ax2.hist(x,label=['HC','AD'],bins=24 - 0*(j%2)*4)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim(left=np.min(x.flatten()), right=np.max(x.flatten()))
        fig2.show()
        #fig2.savefig('fig1_gamma-k{0}-f{1}.pdf'.format(kappa,f))
        
fig.show()




#*** LR test: https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
# H0: GMM. dof = N - 5 (mu1/2,sigma1/2,mix)
# H1: KDEMM. dof = N - 3 (h1/2,mix)
from scipy.stats.distributions import chi2
def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))
LR = likelihood_ratio(L1,L2)
p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
print('p: %.30f' % p)


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





#****** Staging consistency – patients only
kde_ebm_csv_ad = '/Users/noxtoby/Documents/Research/UCLPOND/Projects/201812-KDEEBM/new_manuscript/1-AlzheimersDement/more_files/AD_KDEEBM_stages_res.csv'
kde_ebm_csv_pca = '/Users/noxtoby/Documents/Research/UCLPOND/Projects/201812-KDEEBM/new_manuscript/1-AlzheimersDement/more_files/PCA_KDEEBM_stages_res.csv'
stg = 'Stage'
ebm_ad = kde_ebm_csv_ad
ebm_pca = kde_ebm_csv_pca

# gmm_ebm_csv_ad = '/Users/noxtoby/Documents/Research/UCLPOND/Projects/201812-KDEEBM/new_manuscript/1-AlzheimersDement/more_files/gmmstages_AD.csv'
# gmm_ebm_csv_pca = '/Users/noxtoby/Documents/Research/UCLPOND/Projects/201812-KDEEBM/new_manuscript/1-AlzheimersDement/more_files/gmmstages_PCA.csv'
# stg = 'gmm_stage'
# ebm_ad = gmm_ebm_csv_ad
# ebm_pca = gmm_ebm_csv_pca

df_staging_consistency_AD = pd.read_csv(ebm_ad)
df_staging_consistency_PCA = pd.read_csv(ebm_pca)
#* Patients only
df_staging_consistency_AD = df_staging_consistency_AD.loc[df_staging_consistency_AD.o_diagnosis==1].sort_values(by=['ncf_pid','ncf_vid'])
df_staging_consistency_PCA = df_staging_consistency_PCA.loc[df_staging_consistency_PCA.o_diagnosis==1].sort_values(by=['ncf_pid','ncf_vid'])
#* Longitudinal only
pid_with_followup_AD = df_staging_consistency_AD.ncf_pid[df_staging_consistency_AD.ncf_vid>1].values
pid_with_followup_PCA = df_staging_consistency_PCA.ncf_pid[df_staging_consistency_PCA.ncf_vid>1].values
df_staging_consistency_AD = df_staging_consistency_AD.loc[np.isin(df_staging_consistency_AD.ncf_pid,pid_with_followup_AD)]
df_staging_consistency_PCA = df_staging_consistency_PCA.loc[np.isin(df_staging_consistency_PCA.ncf_pid,pid_with_followup_PCA)]
#* Staging consistency (not counting model uncertainty => To Do)
pid_AD = df_staging_consistency_AD.ncf_pid.unique()
n_followups_AD = np.empty(shape=pid_AD.shape)
n_consistent_AD = np.empty(shape=pid_AD.shape)
bl = df_staging_consistency_AD.ncf_vid.values==1
for i,j in zip(pid_AD,range(len(pid_AD))):
    rowz = i==df_staging_consistency_AD.ncf_pid.values
    stage_bl = df_staging_consistency_AD.loc[rowz & bl,stg].values
    stages = df_staging_consistency_AD.loc[rowz,stg].values
    n_followups_AD[j] = int(len(stages)*(len(stages)-1)/2)
    #* Calculate differences between all combinations of followup
    diffs = [] #np.empty(shape=(1,n_followups_AD[j]))
    for s in range(0,len(stages)):
        diffs += (stages[(s+1):]-stages[s]).tolist()
    n_consistent_AD[j] = sum( [d>=0 for d in diffs] )
print("AD staging consistency: %.2f%% \n  %i of %i followups (all combos) at same or later stage" % (100*sum(n_consistent_AD)/sum(n_followups_AD),sum(n_consistent_AD),sum(n_followups_AD)))

pid_PCA = df_staging_consistency_PCA.ncf_pid.unique()
n_followups_PCA = np.empty(shape=pid_PCA.shape)
n_consistent_PCA = np.empty(shape=pid_PCA.shape)

bl = df_staging_consistency_PCA.ncf_vid.values==1
for i,j in zip(pid_PCA,range(len(pid_PCA))):
    rowz = i==df_staging_consistency_PCA.ncf_pid.values
    stage_bl = df_staging_consistency_PCA.loc[rowz & bl,stg].values
    stages = df_staging_consistency_PCA.loc[rowz,stg].values
    n_followups_PCA[j] = int(len(stages)*(len(stages)-1)/2)
    #* Calculate differences between all combinations of followup
    diffs = [] #np.empty(shape=(1,n_followups_AD[j]))
    for s in range(0,len(stages)):
        diffs += (stages[(s+1):]-stages[s]).tolist()
    n_consistent_PCA[j] = sum( [d>=0 for d in diffs] )
print("PCA staging consistency: %.2f%% \n  %i of %i followups (all combos) at same or later stage" % (100*sum(n_consistent_PCA)/sum(n_followups_PCA),sum(n_consistent_PCA),sum(n_followups_PCA)))
