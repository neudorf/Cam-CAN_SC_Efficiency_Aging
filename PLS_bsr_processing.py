#%%
import numpy as np
import pandas as pd
import math
from pathlib import Path

ROOT_DIR = Path('/media/WDBlue/mcintosh/projects/Cam-CAN/PLS_wrapper_SC/results/')
RESULTS_NAME = 'SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range'
ATLAS = '220'

ROOT_DIR.joinpath(ROOT_DIR,RESULTS_NAME,'figure_files/').mkdir(parents=True)
LUT_FILE = '/media/WDBlue/mcintosh/data/Cam-CAN/data_processing/BrainNet_templates/TVBSchaeferTian220_centres_headers.txt'

MNI_NODE_FILE = 'templates/TVBSchaeferTian218_MNI_node.node'
NEW_MNI_NODE_FILE = 'templates/TVBSchaeferTian218_MNI_node_friendly_names.node'
SCHAEFER_KEY_FILE = 'templates/SchaeferKey.csv'

#%%
def get_input_bsr_triu_path(root_dir,results_name,varname):
    return root_dir.joinpath(f'{results_name}/{results_name}_{varname}_bsr_flat.csv')

def get_input_bsr_matrix_path(root_dir,results_name,varname):
    return root_dir.joinpath(f'{results_name}/{results_name}_{varname}_bsr_matrix.csv')

def get_output_BNV_edge_path(root_dir,results_name,varname,thresh,thresh_direction):
    return root_dir.joinpath(f'{results_name}/figure_files/{results_name}_{varname}_bsr_matrix_thr{thresh}_{thresh_direction}.edge')

def get_ggseg_degree_path(root_dir,results_name,varname,thresh_direction,thresh,deg_thresh):
    return root_dir.joinpath(f'{results_name}/figure_files/{results_name}_{varname}_bsr_{thresh_direction}_thresh{thresh}_degrees{deg_thresh}.csv')

def get_BNV_node_degree_path(root_dir,results_name,varname,thresh_direction,thresh,deg_thresh):
    return root_dir.joinpath(f'{results_name}/figure_files/{results_name}_{varname}_bsr_{thresh_direction}_thresh{thresh}_degrees{deg_thresh}.node')

def flat_upper_tri_regions_n(flat_upper_tri_len):
    """Get number of regions in original matrix from number of cells in upper triangle (only valid for square matrix)
    Derived by solving for x in y = (x)*(x-1)/2 where x is the number of regions and y is the number of cells in the upper triangle
    Parameters
    ----------
    flat_upper_tri_len  :   int
    Returns
    -------
    original_regions_n  :   int
    """
    original_regions_n = 1/2 + math.sqrt(1/4 + 2*flat_upper_tri_len)
    return int(original_regions_n)

def flat_to_square_matrix(file_name,delim=','):
    """Load flattened upper triangle from txt and reshape to square matrix
    Parameters
    ----------
    file_name           :   string
                            file path to flattened upper triangle data
    delim               :   string
                            delimiter string for np.genfromtxt()
    Returns
    -------
    matrix                 :   2-dimensional square ndarray
    """
    triu_flat = np.genfromtxt(file_name,delimiter=delim)
    regions_n = flat_upper_tri_regions_n(triu_flat.size)
    matrix = np.zeros((regions_n,regions_n))
    matrix[np.triu_indices(regions_n,k=1)] = triu_flat.copy()
    matrix += matrix.T
    return matrix

def threshold_matrix(matrix,thresh,thresh_direction):
    """Take in `matrix` and threshold according to `thresh`, in the direction indicated by `thresh_direction`
    Parameters
    ----------
    matrix              :   ndarray
    thresh              :   float, positive (even if using neg thresh_direction, will switch to -1*thresh in function)
    thresh_direction    :   string
                            'pos'   :   positive, will set all values less than thresh to 0.0
                            'neg'   :   negative, will set all values greater than thresh to 0.0
                            'both'  :   positive and negative, will set all values between -thresh and +thresh to 0.0
    Returns
    -------
    matrix_thresholded  :   ndarray
    """
    matrix_thresholded = np.zeros_like(matrix)
    if thresh_direction == 'both':
        matrix_thresholded[np.where(matrix >= thresh)] = np.copy(matrix[np.where(matrix >= thresh)])
        matrix_thresholded[np.where(matrix <= -1*thresh)] = np.copy(matrix[np.where(matrix <= -1*thresh)])
    elif thresh_direction == 'pos':
        matrix_thresholded[np.where(matrix >= thresh)] = np.copy(matrix[np.where(matrix >= thresh)])
    elif thresh_direction == 'neg':
        matrix_thresholded[np.where(matrix <= -1*abs(thresh))] = np.copy(matrix[np.where(matrix <= -1*abs(thresh))])
    return matrix_thresholded

def stride_diag_remove(matrix):
    """This function removes the diagonal and shifts the upper right triangle to the left
    New dimensions will be (regions_n,regions_n-1)
    Parameters
    ----------
    matrix              :   2-D square ndarray
    Returns
    -------
    out                 :   2-D (regions_n-1,regions_n)
    """
    regions_n = matrix.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = matrix.strides
    out = strided(matrix.ravel()[1:], shape=(regions_n-1,regions_n), strides=(s0+s1,s1)).reshape(regions_n,-1)
    return out

def density(matrix):
    """Report the density of a matrix as a proportion of the nonzero upper triangle cells over the total upper triangle cells (excluding diagonal)
    Parameters
    ----------
    matrix              :   2-D square ndarray
    Returns
    -------
    density            :   float
    """
    matrix_triu = np.triu(matrix,k=1)
    nonzero_off_diag_cells_n = np.nonzero(matrix_triu)[0].size
    total_off_diag_cells_n = matrix_triu.size
    density = nonzero_off_diag_cells_n / total_off_diag_cells_n
    return density

def save_BNV_edge(matrix,output_path):
    np.savetxt(output_path, matrix, delimiter='\t', fmt='%f')

def regions_LUT(LUT_path,delim,varname,region_idx):
    """Access region label from look up table (LUT) with headers
    Parameters
    ----------
    LUT_path            :   string, path to file
    delim               :   delimiter to use when accessing `LUT_path`
    region_idx          :   int, index to look up
    Returns
    -------
    region_name         :   string
    """
    LUT_data = pd.read_csv(LUT_path,delimiter=delim)
    region_name = LUT_data[varname][region_idx]
    return region_name

def degree_analysis(matrix,deg_thresh):
    """Prints number of regions in pos and neg networks exceeding deg_thresh
    If a threshold on the cell values is desired this needs to be done before passing to function
    This output is intended to be saved for use with `ggseg` in R
    Parameters
    ----------
    matrix              :   2-D square ndarray
    deg_thresh          :   float
    Returns
    -------
    matrix_thresh_bin_degrees_pos_save, matrix_thresh_bin_degrees_neg_save    :   tuple of 1-D arrays of length matrix.shape[0] with degree values passing deg_thresh or np.nan
    """
    print('-----------------------------------')
    print('degree analysis')
    print('degree threshold:\t',deg_thresh)
    print('-----------------------------------')

    matrix_thresh_bin_pos = np.zeros_like(matrix)
    matrix_thresh_bin_neg = np.zeros_like(matrix)
    matrix_thresh_bin_pos[np.where(matrix > 0.0)] = 1.0
    matrix_thresh_bin_neg[np.where(matrix < 0.0)] = -1.0
    matrix_thresh_bin_degrees_pos = np.sum(matrix_thresh_bin_pos,axis = 1)
    matrix_thresh_bin_degrees_neg = np.sum(matrix_thresh_bin_neg,axis = 1)

    exceed_thresh_idx_pos = np.where(matrix_thresh_bin_degrees_pos>=deg_thresh)
    print(f'pos regions with degree {deg_thresh} or greater:\t',exceed_thresh_idx_pos)
    print(f'pos regions (label) with degree {deg_thresh} or greater:\n',[regions_LUT(LUT_FILE,'\t','label',x) for x in exceed_thresh_idx_pos])
    print(f'pos regions with degree {deg_thresh} or greater(degree value):\t',matrix_thresh_bin_degrees_pos[exceed_thresh_idx_pos])
    print('pos N crossing threshold:\t',matrix_thresh_bin_degrees_pos[exceed_thresh_idx_pos].size)
    exceed_thresh_idx_neg = np.where(matrix_thresh_bin_degrees_neg<=-deg_thresh)
    print(f'neg regions with degree {deg_thresh} or greater:\t',exceed_thresh_idx_pos)
    print(f'neg regions (label) with degree {deg_thresh} or greater:\n',[regions_LUT(LUT_FILE,'\t','label',x) for x in exceed_thresh_idx_neg])
    print(f'neg regions with degree {deg_thresh} or greater(degree value):\t',matrix_thresh_bin_degrees_neg[exceed_thresh_idx_neg])
    print('neg N crossing threshold:\t',matrix_thresh_bin_degrees_neg[exceed_thresh_idx_neg].size)

    matrix_thresh_bin_degrees_pos_save = np.copy(matrix_thresh_bin_degrees_pos)
    matrix_thresh_bin_degrees_pos_save[np.where(matrix_thresh_bin_degrees_pos_save < deg_thresh)] = np.nan
    matrix_thresh_bin_degrees_neg_save = np.copy(matrix_thresh_bin_degrees_neg)
    matrix_thresh_bin_degrees_neg_save[np.where(np.abs(matrix_thresh_bin_degrees_neg_save) < deg_thresh)] = np.nan

    return matrix_thresh_bin_degrees_pos_save, matrix_thresh_bin_degrees_neg_save

def hemisphere_analysis(matrix):
    """Returns the number of nonzero connections in a symmetric matrix stratified by hemispheric type
    Parameters
    ----------
    matrix                                      :   2-D square symmetric ndarray with even N regions
    Returns
    -------
    interhemi_n, lh_intrahemi_n, rh_intrahemi_n :   tuple of ints
    """
    half_regions_n = matrix.shape[0] // 2
    interhemi_matrix = matrix[:half_regions_n,half_regions_n:]
    lh_intrahemi_matrix = np.triu(matrix[:half_regions_n,:half_regions_n],k=1)
    rh_intrahemi_matrix = np.triu(matrix[half_regions_n:,half_regions_n:],k=1)
    interhemi_n = np.nonzero(interhemi_matrix)[0].size
    lh_intrahemi_n = np.nonzero(lh_intrahemi_matrix)[0].size
    rh_intrahemi_n = np.nonzero(rh_intrahemi_matrix)[0].size
    return interhemi_n, lh_intrahemi_n, rh_intrahemi_n

def make_node_file(output_file,degree_array):
    bn_node = pd.read_csv(NEW_MNI_NODE_FILE,names=['x','y','z','colour','size','label'],sep='\t')
    mask = ~np.isnan(degree_array)
    bn_node.loc[mask,'size'] = 2
    bn_node['label'] = bn_node['label'].str.replace('_','-')
    bn_node.to_csv(output_file,sep='\t',header=False, index=False)

#%% Make better looking node names
remove_strings = [
    '17Networks',
    'A_10',
    'A_11',
    'A_1',
    'A_2',
    'A_3',
    'A_4',
    'A_5',
    'A_6',
    'A_7',
    'A_8',
    'A_9',
    '_1',
    '_2',
    '_3',
    '_4',
    '_5',
    '_6',
    '_7',
    '_8',
    '_',
    '-',
    'LH',
    'lh',
    'rh',
    'RH',
    'VisCent',
    'VisPeri',
    #'SomMotA',
    'SomMotB',
    'DorsAttnA',
    'DorsAttnB',
    'SalVentAttnA',
    'SalVentAttnB',
    'LimbicB',
    'LimbicA',
    'ContA',
    'ContB',
    'ContC',
    'DefaultA',
    'DefaultB',
    'DefaultC',
    #'TempPar'
]

node_file = pd.read_csv(MNI_NODE_FILE,sep='\t',names=['x','y','z','colour','size','label'])
for s in remove_strings:
    node_file['label'] = node_file['label'].str.replace(s,'')
key = pd.read_csv(SCHAEFER_KEY_FILE,sep=',')
key_dict = key.set_index('Abbreviation').T.to_dict()
key_dict = {k:v['Full parcel Name'] for (k,v) in key_dict.items()}
for (k,v) in key_dict.items():
    node_file['label'] = node_file['label'].str.replace(k,v)
node_file.to_csv(NEW_MNI_NODE_FILE,sep='\t',index=False,header=False)

#%%
variable_thresh_dict = {'lv1':9.2,
                        'lv2':2.8
                        }

# Threshold bsr matrices, print density info, and save BrainNet Viewer positive and negative edge files
for var in variable_thresh_dict.keys():
    print('=========================')
    print('variable:\t',var)
    print('threshold:\t',variable_thresh_dict[var])
    # Getting square matrix from flat upper tri for bsr values
    bsr_matrix = flat_to_square_matrix(get_input_bsr_triu_path(ROOT_DIR,RESULTS_NAME,var))
    # Print info about density
    print('density:\t',density(threshold_matrix(bsr_matrix,variable_thresh_dict[var],'both')))
    # Saving BrainNet Viewer networks in both positive and negative directions separately
    for thr_dir in ['pos','neg']:
        bsr_matrix_thresholded = threshold_matrix(bsr_matrix,variable_thresh_dict[var],thr_dir)
        save_BNV_edge(bsr_matrix_thresholded,get_output_BNV_edge_path(ROOT_DIR,RESULTS_NAME,var,variable_thresh_dict[var],thr_dir))
    # Making bsr matrix thresholded in both directions for degree_analysis function
    bsr_matrix_thresholded = threshold_matrix(bsr_matrix,variable_thresh_dict[var],'both')
    pos_matrix_thresholded, neg_matrix_thresholded = degree_analysis(bsr_matrix_thresholded,3)
    np.savetxt(get_ggseg_degree_path(ROOT_DIR,RESULTS_NAME,var,'pos',variable_thresh_dict[var],3),pos_matrix_thresholded)
    np.savetxt(get_ggseg_degree_path(ROOT_DIR,RESULTS_NAME,var,'neg',variable_thresh_dict[var],3),-1*neg_matrix_thresholded)
    make_node_file(get_BNV_node_degree_path(ROOT_DIR,RESULTS_NAME,var,'pos',variable_thresh_dict[var],3),pos_matrix_thresholded)
    make_node_file(get_BNV_node_degree_path(ROOT_DIR,RESULTS_NAME,var,'neg',variable_thresh_dict[var],3),-1*neg_matrix_thresholded)
    # Report info about inter/intrahemispheric connections
    print('------------------------')
    print('hemisphere analysis')
    for thr_dir in ['pos','neg']:
        bsr_matrix_thresholded = threshold_matrix(bsr_matrix,variable_thresh_dict[var],thr_dir)
        inter_n, lh_intra_n, rh_intra_n = hemisphere_analysis(bsr_matrix_thresholded)
        total_conn_n = inter_n + lh_intra_n + rh_intra_n
        print('Threshold direction:\t',thr_dir)
        print(f'Interhemispheric connections:\t\tN={inter_n}\t({inter_n/total_conn_n*100}%)')
        print(f'Intrahemispheric connections:\t\tN={lh_intra_n + rh_intra_n}\t({(lh_intra_n+rh_intra_n)/total_conn_n*100}%)')
        print(f'LH Intrahemispheric connections:\tN={lh_intra_n}\t({lh_intra_n/total_conn_n*100}%)')
        print(f'RH Intrahemispheric connections:\tN={rh_intra_n}\t({rh_intra_n/total_conn_n*100}%)')

