#%%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import math

ATLAS = '220'
BEHAVIOURAL_DATA_DIR = Path('../../../data/Cam-CAN/data_processing/behavioural_data')
BEHAVIOURAL_DATA_FILE = BEHAVIOURAL_DATA_DIR.joinpath('behavioural_data.csv')
GT_DATA_DIR = Path('/media/WDBlue/mcintosh/data/Cam-CAN/data_processing/gt_data')
GT_DATA_FILE = GT_DATA_DIR.joinpath('TVBSchaeferTian220_gt_measures.csv')
SC_RAND_DATA_FILE = GT_DATA_DIR.joinpath('TVBSchaeferTian220_SC_randomness_index.csv')
SC_CC_DATA_FILE = GT_DATA_DIR.joinpath('TVBSchaeferTian220_SC_clustering.csv')
SC_SW_DATA_FILE = GT_DATA_DIR.joinpath('TVBSchaeferTian220_SC_smallworldness_sigma.csv')
LE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_local_efficiency.pkl')
NE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_nodal_efficiency_updated.pkl')
SC_DATA_DIR = Path('../../../data/Cam-CAN/data_processing/SC_matrices_consistency_thresholded_0.5_fixed')
SC_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dict.pkl')
SC_DIST_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dist_dict.pkl')
BRAIN_VOL_FILE = Path('../../../data/Cam-CAN/data_processing/freesurfer/brainvol.csv')
RESULTS_ROOT = Path('results')

#%%
def x_y_merge(X_dict,Y_df,vars,filter_vars,matrix=True):
    """Merge `X_dict` dictionary and `Y_df` dataframe based on which keys match 
    subject variable, keeping `vars` and filtering subjects based on those that 
    have values for `filter_vars`.
    Parameters
    ----------
    X_dict          :   dictionary. keys represent subject ids and values represent
                        X data
    Y_df            :   pandas dataframe. contains a `subject` variable with values
                        matching keys in X_dict
    vars            :   list of str. variable names to keep in Y
    filter_vars     :   list of str. variable names to check for missing data when
                        filtering subjects
    matrix          :   bool. Default=True. Treat data differently depending on dimensions
    Returns
    -------
    X, Y, subjects  :   X is 1-D array, Y is 2-D array with first dimension for number
                        of variables second dimension for number of subjects, subjects
                        is list of subject ids
    """
    SC_subjects = list(X_dict.keys())
    SC_subjects_int = [int(x) for x in SC_subjects]
    SC_subjects_df = pd.DataFrame({'subject':SC_subjects_int})
    Y_df_merged = pd.merge(SC_subjects_df,Y_df,on=['subject'],how='left')
    subjects_Y = Y_df_merged[['subject']+filter_vars].dropna()
    subjects_Y = subjects_Y[['subject']+vars]
    subjects = subjects_Y.subject.tolist()
    Y_df = subjects_Y[vars]
    Y = np.array(Y_df)
    if matrix:
        X_matrices = [v for (k,v) in X_dict.items() if int(k) in subjects]
        X = X_matrices.copy()
    else:
        X = np.array([v for (k,v) in X_dict.items() if int(k) in subjects])
    return X, Y, subjects

def make_df_bins(df,bin_width,step_size,var):
    """Makes a list of dataframes that contain a portion of the data based on a range
    of the `var` variable (age was used for the analysis). `bin_width` is the range
    of the `var` to consider, and `step_size` is how much to shift this bin at each
    step
    Parameters
    ----------
    df          : pandas dataframe containing at least a variable `var`
    bin_width   : float. range of values to consider in each bin (e.g., 10 years)
    step_size   : int. number of indices to shift the bin at each step
    var         : str. name of variable to consider
    Returns
    -------
    dfs         : list of pandas dataframes
    """
    dfs = []
    var_min = math.floor(df[var].min())
    var_max = math.ceil(df[var].max())
    var_range = var_max - var_min
    bins_n = int((var_range - bin_width) / step_size) + 1
    for b in range(bins_n):
        bin_min = var_min + b*step_size
        bin_max = bin_min + bin_width
        dfs.append(df.loc[(df[var] >= bin_min) & (df[var] <= bin_max)])

    return dfs

def rolling_analysis(bsrs,bsr_thresh,node_dict,var,filter_vars,bin_width,step_size,figure_prefix,n_resamples=1000):
    """Analyses of rolling bins of data (i.e., rolling correlation analyses).
    Parameters
    ----------
    bsrs            :   1d array. BSR calculated based on node_dict data
    bsr_thresh      :   float. threshold to select significant BSR data
    node_dict       :   dictionary with subject ids as keys. values are data at a
                        node/region level that were used to calculate BSRs
    var             :   str. name of variable for labelling output files
    filter_vars     :   list of str. variable names with which to filter subjects
                        based on whether there are missing values
    bin_width       :   float. range of values to consider in each bin (e.g., 10 years)
    step_size       :   int. number of indices to shift the bin at each step
    figure_prefix   :   str. prefix to use in file name
    n_resamples     :   int. number of resamples/its used in PLS for file name
    Returns
    -------
    neg_cog_corrs_df,pos_cog_corrs_df   :   results saved in pandas dataframe
    """
    bsrs_sig_pos_idx = np.where(bsrs >= bsr_thresh)
    bsrs_sig_neg_idx = np.where(bsrs <= -1*bsr_thresh)

    behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')
    gt_data = pd.read_csv(GT_DATA_FILE,sep=',')
    gt_behaviour_ages = pd.merge(gt_data,behaviour_ages,on='subject',how='left')
    subjects_Y = gt_behaviour_ages[['subject']+filter_vars].dropna()
    subjects = subjects_Y.subject.tolist()

    node_dict_subset = {s:v for (s,v) in node_dict.items() if int(s) in subjects}
    node_dict_subjects = [int(s) for s in node_dict_subset.keys()]

    node_dict_subjects_df = pd.DataFrame({'subject':node_dict_subjects})
    gt_behaviour_ages = pd.merge(node_dict_subjects_df,gt_behaviour_ages,on='subject',how='left')
    subjects = gt_behaviour_ages.subject.to_list()

    values_PLS_sig_pos = []
    values_PLS_sig_neg = []
    for (key,value) in node_dict_subset.items():
        values_PLS_sig_pos.append(np.squeeze(value[bsrs_sig_pos_idx]))
        values_PLS_sig_neg.append(np.squeeze(value[bsrs_sig_neg_idx]))
    array_PLS_sig_pos = np.array(values_PLS_sig_pos)
    array_PLS_sig_neg = np.array(values_PLS_sig_neg)

    sig_df = pd.DataFrame({'subject':subjects,'age':gt_behaviour_ages.age,'CattellTotal':gt_behaviour_ages.CattellTotal,var+'_neg':np.mean(array_PLS_sig_neg,axis=1),var+'_pos':np.mean(array_PLS_sig_pos,axis=1)})

    sig_df_binned = make_df_bins(sig_df,bin_width,step_size,'age')
    
    bootstrap_method = stats.BootstrapMethod(method='BCa', n_resamples=n_resamples)

    neg_cog_cors = []
    bin_N = []
    neg_cog_cors_ci_lower = []
    neg_cog_cors_ci_upper = []
    for df in sig_df_binned:
        bin_N.append(len(df))
        neg_cog_cors.append(pearsonr(df[var+'_neg'],df['CattellTotal'])[0])
        neg_ci = pearsonr(df[var+'_neg'],df['CattellTotal']).confidence_interval(.95,method=bootstrap_method)
        neg_cog_cors_ci_lower.append(neg_ci[0])
        neg_cog_cors_ci_upper.append(neg_ci[1])

    print("bin sizes",bin_N)
    print("mean bin size:",np.mean(bin_N))
    print("std bin size:",np.std(bin_N))
    print("min bin size:",np.min(bin_N))
    print("max bin size:",np.max(bin_N))
    min_age = int(np.min(gt_behaviour_ages.age))
    neg_cog_corrs_df = pd.DataFrame({'age_bin':np.array(range(min_age,min_age+len(neg_cog_cors)))*step_size,var+'_neg_cog_corr':neg_cog_cors,var+'_neg_cog_corr_lower':neg_cog_cors_ci_lower,var+'_neg_cog_corr_upper':neg_cog_cors_ci_upper})
    plt.figure(0)
    sns.scatterplot(neg_cog_corrs_df,x='age_bin',y=var+'_neg_cog_corr')
    plt.savefig(f'figures/{figure_prefix}_inverted_u_neg.png',dpi=600)
    plt.figure(1)
    ax1 = sns.scatterplot(neg_cog_corrs_df,x='age_bin',y=var+'_neg_cog_corr', color=(.224,.604,.694))
    ax1.fill_between(neg_cog_corrs_df.age_bin,neg_cog_corrs_df[var+'_neg_cog_corr_lower'],neg_cog_corrs_df[var+'_neg_cog_corr_upper'],alpha=.2)
    plt.savefig(f'figures/{figure_prefix}_inverted_u_neg_ci.png',dpi=600)

    pos_cog_cors = []
    pos_cog_cors_ci_lower = []
    pos_cog_cors_ci_upper = []
    for df in sig_df_binned:
        pos_cog_cors.append(pearsonr(df[var+'_pos'],df['CattellTotal'])[0])
        pos_ci = pearsonr(df[var+'_pos'],df['CattellTotal']).confidence_interval(.95,method=bootstrap_method)
        pos_cog_cors_ci_lower.append(pos_ci[0])
        pos_cog_cors_ci_upper.append(pos_ci[1])

    pos_cog_corrs_df = pd.DataFrame({'age_bin':np.array(range(min_age,min_age+len(pos_cog_cors)))*step_size,var+'_pos_cog_corr':pos_cog_cors,var+'_pos_cog_corr_lower':pos_cog_cors_ci_lower,var+'_pos_cog_corr_upper':pos_cog_cors_ci_upper})
    plt.figure(2)
    sns.scatterplot(pos_cog_corrs_df,x='age_bin',y=var+'_pos_cog_corr')
    plt.savefig(f'figures/{figure_prefix}_inverted_u_pos.png',dpi=600)
    plt.figure(3)
    ax1 = sns.scatterplot(pos_cog_corrs_df,x='age_bin',y=var+'_pos_cog_corr', color=(.224,.604,.694))
    ax1.fill_between(pos_cog_corrs_df.age_bin,pos_cog_corrs_df[var+'_pos_cog_corr_lower'],pos_cog_corrs_df[var+'_pos_cog_corr_upper'],alpha=.2)
    plt.savefig(f'figures/{figure_prefix}_inverted_u_pos_ci.png',dpi=600)

    return neg_cog_corrs_df,pos_cog_corrs_df

#%%
le_bsrs_file = RESULTS_ROOT.joinpath('SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range/SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range_lv1_bsr.csv')
le_bsrs = np.genfromtxt(le_bsrs_file)
with open(LE_DICT_FILE,'rb') as f:
    SC_le_dict = pickle.load(f)

ne_bsrs_file = RESULTS_ROOT.joinpath('SC_ne_updated_age_CattellTotal_1000_its_350_subs_50_to_150_age_range/SC_ne_updated_age_CattellTotal_1000_its_350_subs_50_to_150_age_range_lv1_bsr.csv')
ne_bsrs = np.genfromtxt(ne_bsrs_file)
with open(NE_DICT_FILE,'rb') as f:
    SC_ne_dict = pickle.load(f)

#%%
neg_df, pos_df = rolling_analysis(ne_bsrs,2,SC_ne_dict,'ne',['age','CattellTotal','Prcsn_PerceptionTest'],10,1,'SC_ne_updated_age_CattellTotal_1000_its_350_subs_50_to_150_age_range')
#%%
neg_df, pos_df = rolling_analysis(le_bsrs,2,SC_le_dict,'le',['age','CattellTotal','Prcsn_PerceptionTest'],10,1,'SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range')

#%%
filter_vars = ['age','CattellTotal','Prcsn_PerceptionTest']
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')
gt_data = pd.read_csv(GT_DATA_FILE,sep=',')
gt_behaviour_ages = pd.merge(gt_data,behaviour_ages,on='subject',how='left')
rand_index = pd.read_csv(SC_RAND_DATA_FILE,sep=',')
gt_behaviour_ages = pd.merge(gt_behaviour_ages,rand_index,on='subject',how='left')
clustering_pd = pd.read_csv(SC_CC_DATA_FILE,sep=',')
gt_behaviour_ages = pd.merge(gt_behaviour_ages,clustering_pd,on='subject',how='left')
smallworld_pd = pd.read_csv(SC_SW_DATA_FILE,sep=',')
gt_behaviour_ages = pd.merge(gt_behaviour_ages,smallworld_pd,on='subject',how='left')
subjects_Y = gt_behaviour_ages[['subject']+filter_vars].dropna()
subjects = subjects_Y.subject.tolist()

#%%
node_dict_subset = {s:v for (s,v) in SC_ne_dict.items() if int(s) in subjects}
node_dict_subjects = [int(s) for s in node_dict_subset.keys()]
node_dict_subjects_df = pd.DataFrame({'subject':node_dict_subjects})
gt_behaviour_ages = pd.merge(node_dict_subjects_df,gt_behaviour_ages,on='subject',how='left')
print(len(gt_behaviour_ages))

#%% Age and CattellTotal plot
age_cog_plot = sns.regplot(data=gt_behaviour_ages,x='age',y='CattellTotal',ci=None, color=(.224,.604,.694))
age_cog_plot.figure.savefig('figures/age_by_CattellTotal.png',dpi=600)
print(pearsonr(gt_behaviour_ages['age'],gt_behaviour_ages['CattellTotal']))

#%%
le_OA_bsrs_file = RESULTS_ROOT.joinpath('SC_le_age_CattellTotal_1000_its_350_subs_50_to_150_age_range/SC_le_age_CattellTotal_1000_its_350_subs_50_to_150_age_range_lv1_bsr.csv')
le_bsrs_OA = np.genfromtxt(le_OA_bsrs_file)
with open(LE_DICT_FILE,'rb') as f:
    SC_le_dict = pickle.load(f)

le_YA_bsrs_file = RESULTS_ROOT.joinpath('SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range/SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range_lv1_bsr.csv')
le_bsrs_YA = np.genfromtxt(le_YA_bsrs_file)

ne_bsrs_YA_file = RESULTS_ROOT.joinpath('SC_ne_age_CattellTotal_1000_its_244_subs_0_to_50_age_range/SC_ne_age_CattellTotal_1000_its_244_subs_0_to_50_age_range_lv1_bsr.csv')
ne_bsrs_YA = np.genfromtxt(ne_bsrs_YA_file)
with open(NE_DICT_FILE,'rb') as f:
    SC_ne_dict = pickle.load(f)

ne_bsrs_OA_file = RESULTS_ROOT.joinpath('SC_ne_age_CattellTotal_1000_its_350_subs_50_to_150_age_range/SC_ne_age_CattellTotal_1000_its_350_subs_50_to_150_age_range_lv1_bsr.csv')
ne_bsrs_OA = np.genfromtxt(ne_bsrs_OA_file)

#%% For reviewer Q
ne_OA_neg_idx = np.where(ne_bsrs_OA < -2.0)
ne_OA_pos_idx = np.where(ne_bsrs_OA > 2.0)
ne_OA_notsigneg_idx = np.where(ne_bsrs_OA > -2.0)

ne_OA_neg_means = [np.mean(v[ne_OA_neg_idx]) for (k,v) in SC_le_dict.items() if int(k) in subjects]
ne_OA_pos_means = [np.mean(v[ne_OA_pos_idx]) for (k,v) in SC_le_dict.items() if int(k) in subjects]
ne_OA_notsigneg_means = [np.mean(v[ne_OA_notsigneg_idx]) for (k,v) in SC_le_dict.items() if int(k) in subjects]

print(stats.ttest_rel(ne_OA_neg_means,ne_OA_pos_means))
print(stats.ttest_rel(ne_OA_neg_means,ne_OA_notsigneg_means))

#%%
gt_behaviour_ages_OA = gt_behaviour_ages.loc[gt_behaviour_ages['age']>50]
gt_behaviour_ages_OA = gt_behaviour_ages_OA.dropna(subset = ['CattellTotal_age_adj'])
OA_subs = gt_behaviour_ages_OA.subject.tolist()
gt_behaviour_ages_YA = gt_behaviour_ages.loc[gt_behaviour_ages['age']<=50]
gt_behaviour_ages_YA = gt_behaviour_ages_YA.dropna(subset = ['CattellTotal_age_adj'])
YA_subs = gt_behaviour_ages_YA.subject.tolist()

le_OA_bad_idx = np.where(le_bsrs_OA < -2.0)
le_OA_bad_means = {k:np.mean(v[le_OA_bad_idx]) for (k,v) in SC_le_dict.items()}
le_OA_bad_means_OA = {k:v for (k,v) in le_OA_bad_means.items() if int(k) in OA_subs}
le_OA_bad_means_YA = {k:v for (k,v) in le_OA_bad_means.items() if int(k) in YA_subs}
gt_behaviour_ages_OA['le_OA_bad_means'] = [v for (k,v) in le_OA_bad_means_OA.items()]
gt_behaviour_ages_YA['le_OA_bad_means'] = [v for (k,v) in le_OA_bad_means_YA.items()]

le_YA_good_idx = np.where(le_bsrs_YA > 2.0)
le_YA_good_means = {k:np.mean(v[le_YA_good_idx]) for (k,v) in SC_le_dict.items()}
le_YA_good_means_OA = {k:v for (k,v) in le_YA_good_means.items() if int(k) in OA_subs}
le_YA_good_means_YA = {k:v for (k,v) in le_YA_good_means.items() if int(k) in YA_subs}
gt_behaviour_ages_OA['le_YA_good_means'] = [v for (k,v) in le_YA_good_means_OA.items()]
gt_behaviour_ages_YA['le_YA_good_means'] = [v for (k,v) in le_YA_good_means_YA.items()]

ne_YA_bad_idx = np.where(ne_bsrs_YA < -2.0)
ne_YA_bad_means = {k:np.mean(v[ne_YA_bad_idx]) for (k,v) in SC_ne_dict.items()}
ne_YA_bad_means_OA = {k:v for (k,v) in ne_YA_bad_means.items() if int(k) in OA_subs}
ne_YA_bad_means_YA = {k:v for (k,v) in ne_YA_bad_means.items() if int(k) in YA_subs}
gt_behaviour_ages_OA['ne_YA_bad_means'] = [v for (k,v) in ne_YA_bad_means_OA.items()]
gt_behaviour_ages_YA['ne_YA_bad_means'] = [v for (k,v) in ne_YA_bad_means_YA.items()]

ne_OA_good_idx = np.where(ne_bsrs_OA > 2.0)
ne_OA_good_means = {k:np.mean(v[ne_OA_good_idx]) for (k,v) in SC_ne_dict.items()}
ne_OA_good_means_OA = {k:v for (k,v) in ne_OA_good_means.items() if int(k) in OA_subs}
ne_OA_good_means_YA = {k:v for (k,v) in ne_OA_good_means.items() if int(k) in YA_subs}
gt_behaviour_ages_OA['ne_OA_good_means'] = [v for (k,v) in ne_OA_good_means_OA.items()]
gt_behaviour_ages_YA['ne_OA_good_means'] = [v for (k,v) in ne_OA_good_means_YA.items()]

#%%
fig, ax = plt.subplots()
sns.regplot(gt_behaviour_ages_OA,x='age',y='le_OA_bad_means',ci=None, color=(.224,.604,.694), ax=ax)
print(pearsonr(gt_behaviour_ages_OA['age'],gt_behaviour_ages_OA['le_OA_bad_means']))

sns.regplot(gt_behaviour_ages_YA,x='age',y='le_OA_bad_means',ci=None, color=(.886,.698,.020), ax=ax)
print(pearsonr(gt_behaviour_ages_YA['age'],gt_behaviour_ages_YA['le_OA_bad_means']))

fig.savefig('figures/SC_le_age_CattellTotal_1000_its_OA_bad_age_regression.png',dpi=600)

#%%
with open(SC_DICT_FILE,'rb') as f:
    SC_dict = pickle.load(f)
with open(SC_DIST_FILE,'rb') as f:
    SC_dist = pickle.load(f)
SC_degree = {k:np.sum(v,axis=0) for (k,v) in SC_dict.items()}
SC_degree_mean = {k:np.mean(v) for (k,v) in SC_degree.items()}
SC_degree_max = {k:np.max(v) for (k,v) in SC_degree.items()}
SC_edge_n = {k:np.where(v>0)[0].shape[0] for (k,v) in SC_dict.items()}
SC_edge_sum = {k:np.sum(v) for (k,v) in SC_dict.items()}
degree_mean_OA = {k:v for (k,v) in SC_degree_mean.items() if int(k) in OA_subs}
degree_max_OA = {k:v for (k,v) in SC_degree_max.items() if int(k) in OA_subs}
edge_sum_OA = {k:v for (k,v) in SC_edge_sum.items() if int(k) in OA_subs}
edge_dist_n_OA = {k:np.where(v>0)[0].shape[0] for (k,v) in SC_dist.items() if int(k) in OA_subs}
edge_dist_OA = {k:np.nansum(v)/edge_dist_n_OA[k] for (k,v) in SC_dist.items() if int(k) in OA_subs}

degree_mean_YA = {k:v for (k,v) in SC_degree_mean.items() if int(k) in YA_subs}
degree_max_YA = {k:v for (k,v) in SC_degree_max.items() if int(k) in YA_subs}
edge_sum_YA = {k:v for (k,v) in SC_edge_sum.items() if int(k) in YA_subs}
edge_dist_n_YA = {k:np.where(v>0)[0].shape[0] for (k,v) in SC_dist.items() if int(k) in YA_subs}
edge_dist_YA = {k:np.nansum(v)/edge_dist_n_YA[k] for (k,v) in SC_dist.items() if int(k) in YA_subs}

gt_behaviour_ages_OA['degree_mean'] = [v for (k,v) in degree_mean_OA.items()]
gt_behaviour_ages_OA['degree_max'] = [v for (k,v) in degree_max_OA.items()]
gt_behaviour_ages_OA['edge_sum'] = [v for (k,v) in edge_sum_OA.items()]
gt_behaviour_ages_OA['edge_dist'] = [v for (k,v) in edge_dist_OA.items()]
gt_behaviour_ages_YA['degree_mean'] = [v for (k,v) in degree_mean_YA.items()]
gt_behaviour_ages_YA['degree_max'] = [v for (k,v) in degree_max_YA.items()]
gt_behaviour_ages_YA['edge_sum'] = [v for (k,v) in edge_sum_YA.items()]
gt_behaviour_ages_YA['edge_dist'] = [v for (k,v) in edge_dist_YA.items()]

#%% demographic info
print('overall age range, mean and SD')
print('range',gt_behaviour_ages.age.min(),'to',gt_behaviour_ages.age.max())
print('mean',gt_behaviour_ages.age.mean())
print('SD',gt_behaviour_ages.age.std())

print('young adult age range, mean and SD')
print('range',gt_behaviour_ages_YA.age.min(),'to',gt_behaviour_ages_YA.age.max())
print('mean',gt_behaviour_ages_YA.age.mean())
print('SD',gt_behaviour_ages_YA.age.std())

print('older adult age range, mean and SD')
print('range',gt_behaviour_ages_OA.age.min(),'to',gt_behaviour_ages_OA.age.max())
print('mean',gt_behaviour_ages_OA.age.mean())
print('SD',gt_behaviour_ages_OA.age.std())

#%% multiple regression analyses without sex
pd.set_option("display.precision", 8)
M1YAa = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_sum',gt_behaviour_ages_YA).fit()
print(M1YAa.summary())
print(M1YAa.summary2().tables[1])

M1OAa = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_sum',gt_behaviour_ages_OA).fit()
print(M1OAa.summary())
print(M1OAa.summary2().tables[1])

#%% 
M1YAb = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_dist',gt_behaviour_ages_YA).fit()
print(M1YAb.summary())
print(M1YAb.summary2().tables[1])

M1OAb = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_dist',gt_behaviour_ages_OA).fit()
print(M1OAb.summary())
print(M1OAb.summary2().tables[1])

#%%
M2YA = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_sum + edge_dist',gt_behaviour_ages_YA).fit()
print(M2YA.summary())
print(M2YA.summary2().tables[1])

M2OA = smf.ols('le_OA_bad_means ~ age + smallworldness_sigma + edge_sum + edge_dist',gt_behaviour_ages_OA).fit()
print(M2OA.summary())
print(M2OA.summary2().tables[1])

#%% multiple regression analyses with sex and smallworldness
#smallworldness_sigma
pd.set_option("display.precision", 10)
M1YAa = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_sum',gt_behaviour_ages_YA).fit()
print(M1YAa.summary())
print(M1YAa.summary2().tables[1])

M1OAa = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_sum',gt_behaviour_ages_OA).fit()
print(M1OAa.summary())
print(M1OAa.summary2().tables[1])

#%%
M1YAb = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_dist',gt_behaviour_ages_YA).fit()
print(M1YAb.summary())
print(M1YAb.summary2().tables[1])

M1OAb = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_dist',gt_behaviour_ages_OA).fit()
print(M1OAb.summary())
print(M1OAb.summary2().tables[1])

#%%
M2YA = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_sum + edge_dist',gt_behaviour_ages_YA).fit()
print(M2YA.summary())
print(M2YA.summary2().tables[1])

M2OA = smf.ols('le_OA_bad_means ~ age + sex + smallworldness_sigma + edge_sum + edge_dist',gt_behaviour_ages_OA).fit()
print(M2OA.summary())
print(M2OA.summary2().tables[1])

#%% Density analyses for reviewer
bsr_thresh_lv1 = 9.2
bsr_thresh_lv2 = 2.8

SC_lv1_bsr = np.genfromtxt(RESULTS_ROOT.joinpath('SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range/SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range_lv1_bsr_matrix.csv'),delimiter=',')
SC_lv2_bsr = np.genfromtxt(RESULTS_ROOT.joinpath('SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range/SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range_lv2_bsr_matrix.csv'),delimiter=',')

SC_lv1_bsr_sig_pos_idx = np.where(SC_lv1_bsr >= bsr_thresh_lv1)
SC_lv1_bsr_sig_neg_idx = np.where(SC_lv1_bsr <= -1*bsr_thresh_lv1)
SC_lv2_bsr_sig_pos_idx = np.where(SC_lv2_bsr >= bsr_thresh_lv2)
SC_lv2_bsr_sig_neg_idx = np.where(SC_lv2_bsr <= -1*bsr_thresh_lv2)

with open(SC_DICT_FILE,'rb') as f:
    temp = pickle.load(f)
    SC_dict = {str(s):temp[str(s)] for s in subjects}
with open(SC_DIST_FILE,'rb') as f:
    temp = pickle.load(f)
    SC_dist_dict = {str(s):temp[str(s)] for s in subjects}

subjects_sex_age_df = gt_behaviour_ages[['subject','sex','age']]
subjects_male = subjects_sex_age_df.loc[subjects_sex_age_df['sex'].isin(['MALE']),'subject'].to_list()
subjects_female = subjects_sex_age_df.loc[subjects_sex_age_df['sex'].isin(['FEMALE']),'subject'].to_list()
subjects_OA = subjects_sex_age_df.loc[subjects_sex_age_df['age'] > 50,'subject'].to_list()
subjects_YA = subjects_sex_age_df.loc[subjects_sex_age_df['age'] < 50,'subject'].to_list()

age = gt_behaviour_ages.loc[gt_behaviour_ages['subject'].isin(subjects)]['age'].to_list()
fluid_intelligence = gt_behaviour_ages.loc[gt_behaviour_ages['subject'].isin(subjects)]['CattellTotal'].to_list()

# GM and WM volume from freesurfer
brainvol_df = pd.read_csv(BRAIN_VOL_FILE,delimiter=',')
cerebral_GM_vol = brainvol_df.loc[brainvol_df['subject'].isin(subjects)]['cerebral_GM_vol'].to_list()
cerebral_WM_vol = brainvol_df.loc[brainvol_df['subject'].isin(subjects)]['cerebral_WM_vol'].to_list()

#%% Demographics
subjects_male_YA_n = len([s for s in subjects_male if s in subjects_YA])
subjects_female_YA_n = len([s for s in subjects_female if s in subjects_YA])
subjects_male_OA_n = len([s for s in subjects_male if s in subjects_OA])
subjects_female_OA_n = len([s for s in subjects_female if s in subjects_OA])

print(f'N Male YA: {subjects_male_YA_n}')
print(f'N Female YA: {subjects_female_YA_n}')
print(f'N Male OA: {subjects_male_OA_n}')
print(f'N Female OA: {subjects_female_OA_n}')

#%% Streamline count: subject differences
streamline_count_dict = {k:np.sum(np.triu(v,1)) for k,v in SC_dict.items()}
streamline_count_np = np.array([v for k,v in streamline_count_dict.items()])
#sns.kdeplot(streamline_count_np)
streamline_count_mean = np.mean(streamline_count_np)
streamline_count_std = np.std(streamline_count_np)
# No outliers found, addressing question about subject differences in streamline count
print(np.where(np.abs(streamline_count_np) > (streamline_count_mean + 3*streamline_count_std)))

#%% Streamline count: sex differences
steamline_count_male = np.array([v for k,v in streamline_count_dict.items() if int(k) in subjects_male])
steamline_count_female = np.array([v for k,v in streamline_count_dict.items() if int(k) in subjects_female])

print(ttest_ind(steamline_count_male,steamline_count_female))
print(f'male mean: {np.mean(steamline_count_male)}')
print(f'male SD: {np.std(steamline_count_male)}')
print(f'female mean: {np.mean(steamline_count_female)}')
print(f'female SD: {np.std(steamline_count_female)}')

sns.kdeplot(steamline_count_male)
sns.kdeplot(steamline_count_female)

#%% Streamline count: relationship with age and fluid intelligence
# Model not significant. No sig effects.
model_df = pd.DataFrame({'streamline_count':streamline_count_np,'age':age,'fluid_intelligence':fluid_intelligence})
M1 = smf.ols('streamline_count ~ age + fluid_intelligence',model_df).fit()
print(M1.summary())
print(M1.summary2().tables[1])

#%% Streamline distance comparison between SC streamline count LV sig bsr connections
# These distances are in mm
SC_dist_lv1_sig_pos = [np.mean(v[SC_lv1_bsr_sig_pos_idx]) for k,v in SC_dist_dict.items()]
SC_dist_lv1_sig_neg = [np.mean(v[SC_lv1_bsr_sig_neg_idx]) for k,v in SC_dist_dict.items()]
SC_dist_lv2_sig_pos = [np.mean(v[SC_lv2_bsr_sig_pos_idx]) for k,v in SC_dist_dict.items()]
SC_dist_lv2_sig_neg = [np.mean(v[SC_lv2_bsr_sig_neg_idx]) for k,v in SC_dist_dict.items()]

print('LV2 pos vs neg:')
print(stats.ttest_rel(SC_dist_lv2_sig_pos,SC_dist_lv2_sig_neg))
print(f'LV2 pos mean: {np.mean(SC_dist_lv2_sig_pos)}')
print(f'LV2 pos SD: {np.std(SC_dist_lv2_sig_pos)}')
print(f'LV2 neg mean: {np.mean(SC_dist_lv2_sig_neg)}')
print(f'LV2 neg SD: {np.std(SC_dist_lv2_sig_neg)}')
print('LV2 pos vs LV1 pos')
print(stats.ttest_rel(SC_dist_lv2_sig_pos,SC_dist_lv1_sig_pos))
print(f'LV2 pos mean: {np.mean(SC_dist_lv2_sig_pos)}')
print(f'LV2 pos SD: {np.std(SC_dist_lv2_sig_pos)}')
print(f'LV1 pos mean: {np.mean(SC_dist_lv1_sig_pos)}')
print(f'LV1 pos SD: {np.std(SC_dist_lv1_sig_pos)}')

#%% Brain volume
print('WM',pearsonr(streamline_count_np.tolist(),cerebral_WM_vol))
print('GM',pearsonr(streamline_count_np.tolist(),cerebral_GM_vol))