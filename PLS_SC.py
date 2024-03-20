# %%
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from plotnine import *
import copy
import math

# https://github.com/neudorf/PLS_wrapper
from PLS_wrapper import pls

ATLAS = '220'
BEHAVIOURAL_DATA_DIR = Path('../../../data/Cam-CAN/data_processing/behavioural_data')
BEHAVIOURAL_DATA_FILE = BEHAVIOURAL_DATA_DIR.joinpath('behavioural_data.csv')
GT_DATA_DIR = Path('../../../data/Cam-CAN/data_processing/gt_data')
GT_DATA_FILE = GT_DATA_DIR.joinpath('TVBSchaeferTian220_gt_measures.csv')
LE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_local_efficiency.pkl')
NE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_nodal_efficiency_updated.pkl')
DNE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_diff_nodal_efficiency.pkl')
DLE_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_diff_local_efficiency.pkl')
DEG_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_degree.pkl')
PR_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_pagerank.pkl')
BC_DICT_FILE = GT_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}_SC_betweenness_centrality.pkl')
SC_DATA_DIR = Path('../../../data/Cam-CAN/data_processing/SC_matrices_consistency_thresholded_0.5_fixed')
SC_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dict.pkl')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)
ITS = 1000

# %%
def matrix_to_flat_triu(matrix):
    """Take numpy matrix and get flat upper triangle
    Parameters
    ----------
    matrix              :   ndarray
                            square adjacency matrix
    Returns
    -------
    triu                :   1-D ndarray with all cells in upper triangle
    """
    regions_n = matrix.shape[0]
    triu = matrix[np.triu_indices(regions_n,k=1)]
    return triu

def flat_upper_tri_regions_n(flat_upper_tri_len):
    """Get number of regions in original matrix from number of cells in upper triangle (only valid for square matrix)
    Derived by solving for x in y = (x)*(x-1)/2 where x is the number of regions and y is the number of cells in the upper triangle
    Parameters
    ----------
    flat_upper_tri_len  :   int
                            must be positive
    Returns
    -------
    original_regions_n:   int
    """
    original_regions_n = 1/2 + math.sqrt(1/4 + 2*flat_upper_tri_len)
    return int(original_regions_n)

def flat_to_square_matrix(triu_flat):
    """Load flattened upper triangle from txt and reshape to square matrix
    Parameters
    ----------
    triu_flat           :   1-D ndarray
                            flattened upper triangle data
    Returns
    -------
    matrix                 :   2-dimensional square ndarray
    """
    regions_n = flat_upper_tri_regions_n(triu_flat.size)
    matrix = np.zeros((regions_n,regions_n))
    matrix[np.triu_indices(regions_n,k=1)] = triu_flat.copy()
    matrix += matrix.T
    return matrix

def pls_x_y_merge(X_dict,Y_df,vars,filter_vars,matrix=True):
    """Merge `X_dict` dictionary and `Y_df` dataframe based on which keys match 
    subject variable, keeping `vars` and filtering subjects based on those that 
    have values for `filter_vars`. If matrix set to true then flatten to upper triangular
    values.
    Parameters
    ----------
    X_dict      :   dictionary. keys represent subject ids and values represent
                    X data
    Y_df        :   pandas dataframe. contains a `subject` variable with values
                    matching keys in X_dict
    vars        :   list of str. variable names to keep in Y
    filter_vars :   list of str. variable names to check for missing data when
                    filtering subjects
    matrix      :   bool. Default=True. Whether to take upper triange of matrix or
                    leave data as is if already 1-D array
    Returns
    -------
    X, Y        :   X is 1-D array, Y is 2-D array with first dimension for number
                     of variables second dimension for number of subjects 
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
        X_trius = np.array([matrix_to_flat_triu(mat) for mat in X_matrices])
        X = X_trius.copy()
    else:
        X = np.array([v for (k,v) in X_dict.items() if int(k) in subjects])
    return X, Y

def pls_process_results(res,vars,age_range,X_name,its,subjects_n,output_dir,printing=True,matrix=True):
    """Takes results from PLS analysis and produces some important outputs.
    Prints behavioural correlation plot, saves BSR values as a matrix, saves permuation
    p value, percent covariance, and saves model. 
    Parameters
    ----------
    res             :   Dict2Obj results from `PLS_wrapper` package `pls` function.
    vars            :   list of str. variable names used in analysis
    age_range       :   list of str. min and max age values in sample
    its             :   int. number of iterations used for permutation and bootstrap
                        resampling by PLS
    subjects_n      :   int. number of subjects in sample
    output_dir      :   str or patlib Path. path to output directory to save outputs
    printing        :   bool. Default=True. Whether to print outputs as they are calculated
    matrix          :   bool. Default=True. Whether to save flattened BSR values
                        as full symmetric matrix.
    Returns
    -------
    None
    """
    res_cp = copy.deepcopy(res)
    Y_name = '_'.join(vars)
    output_dir = output_dir.joinpath(f'{X_name}_{Y_name}_{its}_its_{subjects_n}_subs_{age_range[0]}_to_{age_range[1]}_age_range')
    output_dir.mkdir(exist_ok=True)

    lvs_n = len(vars)
    if lvs_n == 1:
        res_cp.s = res_cp.s[None]
        res_cp.boot_result.llcorr = res_cp.boot_result.llcorr[None]
        res_cp.boot_result.ulcorr = res_cp.boot_result.ulcorr[None]
        res_cp.boot_result.orig_corr = res_cp.boot_result.orig_corr[None]
        res_cp.boot_result.compare_u[None]

    behav_corrs = []
    ll_corrs = []
    ul_corrs = []
    percent_covs = []
    bsrs = []
    behav_corr_plots = []
    for lv in range(lvs_n):
        percent_covs.append(res_cp.s[lv]**2/sum(res_cp.s**2)*100)
        if len(vars) == 1:
            behav_corrs.append(res_cp.boot_result.orig_corr)
            ll_corrs.append(res_cp.boot_result.llcorr)
            ul_corrs.append(res_cp.boot_result.ulcorr)
        else:
            behav_corrs.append(res_cp.boot_result.orig_corr[:,lv])
            ll_corrs.append(res_cp.boot_result.llcorr[:,lv])
            ul_corrs.append(res_cp.boot_result.ulcorr[:,lv])
        bsrs.append(res_cp.boot_result.compare_u[:,lv])
        print(behav_corrs[lv])
        print(ll_corrs[lv])
        print(ul_corrs[lv])
        print(len(vars))

        behav_corr_df = pd.DataFrame({'behav_corr':behav_corrs[lv],
                                    'll_corr':ll_corrs[lv],
                                    'ul_corr':ul_corrs[lv],
                                    'vars':[v.capitalize() for v in vars]})
        behav_corr_plots.append((
                                ggplot(behav_corr_df,aes(x='vars',y='behav_corr',fill='vars'))
                                + geom_bar(stat="identity", position=position_dodge(), show_legend=False)
                                + geom_errorbar(aes(ymin=ll_corrs[lv], ymax=ul_corrs[lv]))
                                + labs(x='Dependent Variables',y='Behavioural Correlation')
                                + theme_classic()
                                ))

        file_prefix = f'{X_name}_{Y_name}_{its}_its_{subjects_n}_subs_{age_range[0]}_to_{age_range[1]}_age_range'
        file_prefix_lv = f'{file_prefix}_lv{lv+1}'
        behav_corr_plots[lv].save(output_dir.joinpath(f'{file_prefix_lv}_behav_corr.png'))
        if matrix:
            bsr_matrix = flat_to_square_matrix(bsrs[lv])
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr_matrix.csv'),bsr_matrix,delimiter=',')
            bsr_matrix_flat = matrix_to_flat_triu(bsr_matrix)
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr_flat.csv'),bsr_matrix_flat)
        else:
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr.csv'),bsrs[lv])
        behav_corr_df.to_csv(output_dir.joinpath(f'{file_prefix_lv}_behav_corr.csv'),index=False)
    percent_covs_df = pd.DataFrame({'percent_cov':percent_covs,'lv':[x+1 for x in list(range(lvs_n))]})
    percent_covs_df.to_csv(output_dir.joinpath(f'{file_prefix}_percent_cov.csv'),index=False)
    if len(vars) == 1:
        perm_p = res_cp.perm_result.sprob
    else:
        perm_p = res_cp.perm_result.sprob[:,0]
    perm_p_df = pd.DataFrame({'perm_p':perm_p,'lv':[x+1 for x in list(range(lvs_n))]})
    perm_p_df.to_csv(output_dir.joinpath(f'{file_prefix}_perm_p.csv'),index=False)
    pls.save_pls_model(str(output_dir.joinpath(f'{file_prefix}_model.mat')),res)

    if printing:
        print(Y_name)
        for lv in range(lvs_n):
            print(f'Permutation p:\t\t{perm_p[lv]:.6f}')
            print(f'Percent Covariance:\t{percent_covs[lv][0]:.6f}')
            print(behav_corr_plots[lv])

# %%
with open(SC_DICT_FILE,'rb') as f:
    SC_dict = pickle.load(f)
with open(LE_DICT_FILE,'rb') as f:
    SC_le_dict = pickle.load(f)
with open(NE_DICT_FILE,'rb') as f:
    SC_ne_dict = pickle.load(f)
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')
gt_behaviour_ages = behaviour_ages
gt_data = pd.read_csv(GT_DATA_FILE,sep=',')
gt_behaviour_ages = pd.merge(behaviour_ages,gt_data,on='subject',how='left')

# %%
age_ranges = [
    [0,150],
    [0,50],
    [50,150]
]

#Filtering to subjects with all of these variables' data for consistency with other analyses
filter_variables = ['age','CattellTotal','Prcsn_PerceptionTest']

data_dicts = {
    'SC':SC_dict,
    'SC_le':SC_le_dict,
    'SC_ne':SC_ne_dict,
}

#%% Just saving the data for use in matlab (for reliability analyses)
variable_combos = [['age','CattellTotal']]

for variables in variable_combos:
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            age_filter = (gt_behaviour_ages['age'] > age_range[0]) & (gt_behaviour_ages['age'] <= age_range[1])
            gt_behaviour_ages_filtered = gt_behaviour_ages.loc[age_filter]
            if data_dict_name == 'SC':
                matrix = True
            else:
                matrix = False
            X, Y = pls_x_y_merge(data_dict_value,gt_behaviour_ages_filtered,variables,filter_variables,matrix=matrix)
            Y_name = '_'.join(variables)
            subjects_n = X.shape[0]
            file_prefix = f'{data_dict_name}_{Y_name}_{ITS}_its_{subjects_n}_subs_{age_range[0]}_to_{age_range[1]}_age_range'
            np.savetxt(f'PLS_reliability/{file_prefix}_X.csv', X, delimiter=',')
            np.savetxt(f'PLS_reliability/{file_prefix}_Y_age_CattellTotal.csv', Y, delimiter=',')

# %% PLS analysis now
            
variable_combos = [
    ['age','CattellTotal'],
]

for variables in variable_combos:
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            age_filter = (gt_behaviour_ages['age'] > age_range[0]) & (gt_behaviour_ages['age'] <= age_range[1])
            gt_behaviour_ages_filtered = gt_behaviour_ages.loc[age_filter]
            if data_dict_name == 'SC':
                matrix = True
            else:
                matrix = False
            X, Y = pls_x_y_merge(data_dict_value,gt_behaviour_ages_filtered,variables,filter_variables,matrix=matrix)
            print(Y.shape)
            res = pls.pls_analysis(X,Y.shape[0],1,Y,
                                    num_perm=ITS,
                                    num_boot=ITS,
                                    make_script=False)

            pls_process_results(res,variables,age_range,data_dict_name,ITS,Y.shape[0],OUTPUT_DIR,printing=True,matrix=matrix)

#%% Edge files for BrainNet Viewer
LV2_thresh = 2.8
results_name = 'SC_age_CattellTotal_1000_its_596_subs_0_to_150_age_range'
edge_file = np.genfromtxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv2_bsr_flat.csv'),delimiter=',')
edge_file = flat_to_square_matrix(edge_file)
edge_file[np.where(edge_file<LV2_thresh)] = 0
np.savetxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv2_bsr_matrix.edge'),edge_file,delimiter='\t', fmt='%f')

edge_file_neg = np.genfromtxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv2_bsr_flat.csv'),delimiter=',')
edge_file_neg = flat_to_square_matrix(edge_file_neg)
print(np.where(edge_file_neg<-LV2_thresh)[0].shape)
edge_file_neg[np.where(edge_file_neg>-1*LV2_thresh)] = 0
edge_file_neg[np.where(edge_file_neg>-1*LV2_thresh)] = -1 * edge_file_neg[np.where(edge_file_neg>-1*LV2_thresh)]
np.savetxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv2_bsr_neg_matrix.edge'),edge_file_neg,delimiter='\t', fmt='%f')

LV1_thresh = 9.2
edge_file = np.genfromtxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv1_bsr_flat.csv'),delimiter=',')
edge_file = flat_to_square_matrix(edge_file)
edge_file[np.where(edge_file<LV1_thresh)] = 0
np.savetxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv1_bsr_matrix.edge'),edge_file,delimiter='\t', fmt='%f')

edge_file_neg = np.genfromtxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv1_bsr_flat.csv'),delimiter=',')
edge_file_neg = flat_to_square_matrix(edge_file_neg)
print(np.where(edge_file_neg<-1*LV1_thresh)[0].shape)
edge_file_neg[np.where(edge_file_neg>-1*LV1_thresh)] = 0
edge_file_neg[np.where(edge_file_neg>-1*LV1_thresh)] = -1 * edge_file_neg[np.where(edge_file_neg>-1*LV1_thresh)]
np.savetxt(OUTPUT_DIR.joinpath(f'{results_name}/{results_name}_lv1_bsr_neg_matrix.edge'),edge_file_neg,delimiter='\t', fmt='%f')

# %% Sex analyses SC
# Each of these sex reanalyses find 1 sig LV that finds no pattern of diff between sexes.
# (e.g., orig_corr -> Male: +Age -Cog; Female: +Age -Cog)
sex_filter_male = gt_behaviour_ages['sex'] == 'MALE'
gt_behaviour_ages_male = gt_behaviour_ages.loc[sex_filter_male]
age_filter_female = gt_behaviour_ages['sex'] == 'FEMALE'
gt_behaviour_ages_female = gt_behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['SC'],gt_behaviour_ages_male,['age','CattellTotal'],['age','CattellTotal'],matrix=True)
X_female, Y_female = pls_x_y_merge(data_dicts['SC'],gt_behaviour_ages_female,['age','CattellTotal'],['age','CattellTotal'],matrix=True)

Y = np.append(Y_male,Y_female,axis=0)

res = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

# %% Sex analyses SC_le
sex_filter_male = gt_behaviour_ages['sex'] == 'MALE'
gt_behaviour_ages_male = gt_behaviour_ages.loc[sex_filter_male]
age_filter_female = gt_behaviour_ages['sex'] == 'FEMALE'
gt_behaviour_ages_female = gt_behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['SC_le'],gt_behaviour_ages_male,['age','CattellTotal'],['age','CattellTotal'],matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['SC_le'],gt_behaviour_ages_female,['age','CattellTotal'],['age','CattellTotal'],matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_SC_le = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

# %% Sex analyses SC_ne
sex_filter_male = gt_behaviour_ages['sex'] == 'MALE'
gt_behaviour_ages_male = gt_behaviour_ages.loc[sex_filter_male]
age_filter_female = gt_behaviour_ages['sex'] == 'FEMALE'
gt_behaviour_ages_female = gt_behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['SC_ne_updated'],gt_behaviour_ages_male,['age','CattellTotal'],['age','CattellTotal'],matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['SC_ne_updated'],gt_behaviour_ages_female,['age','CattellTotal'],['age','CattellTotal'],matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_SC_ne = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)