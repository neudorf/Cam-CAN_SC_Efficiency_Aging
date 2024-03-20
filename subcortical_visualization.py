#%%
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage
from PIL import Image
from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

its = 1000
YA_n = 244
OA_n = 350
all_n = 594
var_names = 'SC_ne_age_CattellTotal'

ATLAS_DIR = Path('../../../data/Cam-CAN/atlas')
ATLAS_FILE = ATLAS_DIR.joinpath('TVB_SchaeferTian_fixed_218.nii.gz')
DATA_FILE_YA = Path(f'results/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr.csv')
DATA_FILE_OA = Path(f'results/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr.csv')
DATA_FILE_ALL = Path(f'results/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr.csv')
OUTPUT_FILE_NEG_YA = Path(f'results/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_neg_subcort.nii.gz')
OUTPUT_FILE_POS_YA = Path(f'results/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_pos_subcort.nii.gz')
OUTPUT_FILE_NEG_OA = Path(f'results/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_neg_subcort.nii.gz')
OUTPUT_FILE_POS_OA = Path(f'results/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_pos_subcort.nii.gz')
OUTPUT_FILE_NEG_ALL = Path(f'results/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_neg_subcort.nii.gz')
OUTPUT_FILE_POS_ALL = Path(f'results/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_pos_subcort.nii.gz')

BG_NIFTI_FILE = Path('/usr/local/fsl/data/standard/MNI152_T1_0.5mm.nii.gz')
SUBCORT_ATLAS_NIFTI_FILE = Path('/example/data/path/TVB_SchaeferTian_218_subcort.nii.gz')

FIGURE_FILE_YA_LH = Path(f'figures/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_subcort_LH.png')
FIGURE_FILE_YA_RH = Path(f'figures/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_subcort_RH.png')
FIGURE_FILE_YA_AXIAL = Path(f'figures/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_subcort_axial.png')
FIGURE_FILE_YA_COMBINED = Path(f'figures/{var_names}_{its}_its_{YA_n}_subs_0_to_50_age_range_lv1_bsr_subcort_combined.png')
FIGURE_FILE_OA_LH = Path(f'figures/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_subcort_LH.png')
FIGURE_FILE_OA_RH = Path(f'figures/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_subcort_RH.png')
FIGURE_FILE_OA_AXIAL = Path(f'figures/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_subcort_axial.png')
FIGURE_FILE_OA_COMBINED = Path(f'figures/{var_names}_{its}_its_{OA_n}_subs_50_to_150_age_range_lv1_bsr_subcort_combined.png')
FIGURE_FILE_ALL_LH = Path(f'figures/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_subcort_LH.png')
FIGURE_FILE_ALL_RH = Path(f'figures/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_subcort_RH.png')
FIGURE_FILE_ALL_AXIAL = Path(f'figures/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_subcort_axial.png')
FIGURE_FILE_ALL_COMBINED = Path(f'figures/{var_names}_{its}_its_{all_n}_subs_0_to_150_age_range_lv1_bsr_subcort_combined.png')

def save_ST_nifti(atlas_file,data,output_file,regions=None,subcortical=False):
    """Saves nifti file after assigning values in data to ST atlas
    Parameters
    ----------
    atlas_file  :   string. Path to atlas nifti file
    data        :   1-D array. Data to assign to atlas, with index equal to
                    region number - 1
    output_file :   string. Path to output nifti file
    regions     :   list. Default=None. Regions to assign data to. If none are
                    given then will assign to all ROIs.
    subcortical :   bool. Default=False. Easy way to assign only to subcortical
                    regions by setting to True.
    Returns
    -------
    None
    """
    if not regions:
        regions = list(range(len(data)))
    if subcortical:
        regions = list(range(100,109)) + list(range(209,218))
    atlas_img = nib.load(atlas_file)
    atlas_affine = atlas_img.affine
    atlas_data = atlas_img.get_fdata()

    output_data = np.zeros_like(atlas_data)

    for r in regions:
        atlas_region_idx = np.where(atlas_data == float(r + 1))
        output_data[atlas_region_idx] = data[r]

    output_img = nib.Nifti1Image(output_data, atlas_affine)
    nib.save(output_img, output_file)

#%%
def threshold_data(data_file,thresh_direction,thresh_value):
    """Theshold data by setting values to zero if above (neg direction) or below
    (pos direction) threshold
    Parameters
    ----------
    data_file           :   str or pathlib Path. file here is csv which will be 
                            loaded as ndarray
    thresh_direction    :   str. 'pos' or 'neg'. 'pos' will set zeros below `thresh_value`
                            to 0, 'neg will set zeros above `thresh_value` to 0
    thresh_value        :   float. value at which to threshold
    Returns
    -------
    data                :   ndarray in same dimensions as data in `data_file`
    """
    data = np.genfromtxt(data_file)
    if thresh_direction == 'neg':
        data[np.where(data >= thresh_value)] = 0.0
        data *= -1.0
    elif thresh_direction == 'pos':
        data[np.where(data <= thresh_value)] = 0.0
    return data

data_YA_neg = threshold_data(DATA_FILE_YA,'neg',-2.0)
data_YA_pos = threshold_data(DATA_FILE_YA,'pos',2.0)
save_ST_nifti(ATLAS_FILE,data_YA_neg,OUTPUT_FILE_NEG_YA,subcortical=True)
data_YA_neg_max = np.max(data_YA_neg)
print('YA_neg_max',data_YA_neg_max)
save_ST_nifti(ATLAS_FILE,data_YA_pos,OUTPUT_FILE_POS_YA,subcortical=True)
data_YA_pos_max = np.max(data_YA_pos)
print('YA_pos_max',data_YA_pos_max)
data_YA_max = np.max([data_YA_neg_max,data_YA_pos_max])

data_OA_neg = threshold_data(DATA_FILE_OA,'neg',-2.0)
data_OA_pos = threshold_data(DATA_FILE_OA,'pos',2.0)
save_ST_nifti(ATLAS_FILE,data_OA_neg,OUTPUT_FILE_NEG_OA,subcortical=True)
data_OA_neg_max = np.max(data_OA_neg)
print('OA_neg_max',data_OA_neg_max)
save_ST_nifti(ATLAS_FILE,data_OA_pos,OUTPUT_FILE_POS_OA,subcortical=True)
data_OA_pos_max = np.max(data_OA_pos)
print('OA_pos_max',data_OA_pos_max)
data_OA_max = np.max([data_OA_neg_max,data_OA_pos_max])

data_all_neg = threshold_data(DATA_FILE_ALL,'neg',-2.0)
data_all_pos = threshold_data(DATA_FILE_ALL,'pos',2.0)
save_ST_nifti(ATLAS_FILE,data_all_neg,OUTPUT_FILE_NEG_ALL,subcortical=True)
data_all_neg_max = np.max(data_all_neg)
print('all_neg_max',data_all_neg_max)
save_ST_nifti(ATLAS_FILE,data_all_pos,OUTPUT_FILE_POS_ALL,subcortical=True)
data_all_pos_max = np.max(data_all_pos)
print('all_pos_max',data_all_pos_max)
data_all_max = np.max([data_all_neg_max,data_all_pos_max])

#%%
# Nilearn plotting method

BG_img = nib.load(BG_NIFTI_FILE)
subcort_overlay = nib.load(SUBCORT_ATLAS_NIFTI_FILE)
YA_neg_img = nib.load(OUTPUT_FILE_NEG_YA)
YA_pos_img = nib.load(OUTPUT_FILE_POS_YA)
OA_neg_img = nib.load(OUTPUT_FILE_NEG_OA)
OA_pos_img = nib.load(OUTPUT_FILE_POS_OA)
all_neg_img = nib.load(OUTPUT_FILE_NEG_ALL)
all_pos_img = nib.load(OUTPUT_FILE_POS_ALL)

#%%
greyscale_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(1,1,1)],N=1000)
black_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(0,0,0)],N=1000)
yellow_red_cmap = LinearSegmentedColormap.from_list('greyscale', [(.906,.761,.106),(.941,.094,0)],N=1000)
green_blue_cmap = LinearSegmentedColormap.from_list('greyscale', [(.698,.757,.463),(.224,.604,.694)],N=1000)

def make_subcort_images(BG_img,subcort_overlay,neg_img,pos_img,img_thresh,vmax_neg,vmax_pos,zoom_scale):
    """Makes zoomed in subcortical images applying a black subcortical overlay onto
    the background image before adding positively valued data and negatively valued
    (abs value) data as overlays. Zoom scale set as 3 when using here but would need
    to be changed if resolution of background image changed.
    Parameters
    ----------
    BG_img              : nib img. Background structural image (e.g., MNI)
    subcort_overlay     : nib img. subcortical data to be set black for background
    neg_img             : nib img. negative valued data to overlay (should be abs)
    pos_img             : nib img. positive valued data to overlay
    img_thresh          : float. threshold to use for visualization
    vmax_neg            : float. threshold for vmax of negative overlay
    vmax_pos            : float. threshold for vmax of positive overlay
    zoom_scale          : float. zoom factor used by scipy.ndimage.zoom (set 3)
    Returns
    -------
    img_np_zoomed_LH, img_np_zoomed_RH, img_np_zoomed_axial     : ndarrays of images
    """

    def zoom(img_file,zoom_scale):
        tmp_img = Path('/tmp/figure.png')
        img_file.savefig(tmp_img)
        img = Image.open(tmp_img)
        img_np = np.array(img.getdata()).reshape(img.size[1],img.size[0], 4)
        img_np_zoomed = ndimage.zoom(img_np, (zoom_scale,zoom_scale,1), order=0)
        return img_np_zoomed

    def subcort_plotting(display_mode,cut_coords):
        subcort_img = plotting.plot_img(    BG_img, 
                                            display_mode=display_mode,
                                            cut_coords=cut_coords,
                                            draw_cross=False,
                                            cmap=greyscale_cmap,
                                            figure=1,
                                            annotate=False,
                                            )

        subcort_img.add_overlay(            subcort_overlay,
                                            cmap=black_cmap,
                                            threshold=1.0,
                                            )
        if vmax_pos > 0:
            subcort_img.add_overlay(        pos_img,
                                            vmin=2,
                                            vmax=vmax_pos,
                                            cmap=yellow_red_cmap,
                                            threshold=img_thresh,                 
                                            )
        if vmax_neg > 0:
            subcort_img.add_overlay(        neg_img, 
                                            vmin=2,
                                            vmax=vmax_neg,
                                            cmap=green_blue_cmap,
                                            threshold=img_thresh,                 
                                            )
        return subcort_img
    
    LH_sag_coord = -28.0
    RH_sag_coord = 30.0
    axial_coord = 0.0

    subcort_img_LH = subcort_plotting('x',[LH_sag_coord])

    xxadj = -15
    xyadj = -9
    xbase = 35
    subcort_img_LH.axes[LH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
    subcort_img_LH.axes[LH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
    img_np_zoomed_LH = zoom(subcort_img_LH,zoom_scale)[:,:,:-1]

    subcort_img_RH = subcort_plotting('x',[RH_sag_coord])

    xxadj = -15
    xyadj = -9
    xbase = 35
    subcort_img_RH.axes[RH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
    subcort_img_RH.axes[RH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
    img_np_zoomed_RH = zoom(subcort_img_RH,zoom_scale)[:,:,:-1]
    
    subcort_img_axial = subcort_plotting('z',[axial_coord])

    zxadj = 1
    zyadj = -10
    zbase = 50
    subcort_img_axial.axes[axial_coord].ax.set_xlim(-zbase+zxadj,zbase+zxadj)
    subcort_img_axial.axes[axial_coord].ax.set_ylim(-zbase+zyadj,zbase+zyadj)
    img_np_zoomed_axial = zoom(subcort_img_axial,zoom_scale)[:,:,:-1]
    
    return img_np_zoomed_LH, img_np_zoomed_RH, img_np_zoomed_axial

#%%
img_YA_LH, img_YA_RH, img_YA_axial = make_subcort_images(BG_img, subcort_overlay, YA_neg_img, YA_pos_img,2.0,data_YA_max,data_YA_max,3)
Image.fromarray(img_YA_LH.astype('uint8'), 'RGB').save(FIGURE_FILE_YA_LH)
Image.fromarray(img_YA_RH.astype('uint8'), 'RGB').save(FIGURE_FILE_YA_RH)
Image.fromarray(img_YA_axial.astype('uint8'), 'RGB').save(FIGURE_FILE_YA_AXIAL)
img_OA_LH, img_OA_RH, img_OA_axial = make_subcort_images(BG_img, subcort_overlay, OA_neg_img, OA_pos_img,2.0,data_OA_max,data_OA_max,3)
Image.fromarray(img_OA_LH.astype('uint8'), 'RGB').save(FIGURE_FILE_OA_LH)
Image.fromarray(img_OA_RH.astype('uint8'), 'RGB').save(FIGURE_FILE_OA_RH)
Image.fromarray(img_OA_axial.astype('uint8'), 'RGB').save(FIGURE_FILE_OA_AXIAL)
img_all_LH, img_all_RH, img_all_axial = make_subcort_images(BG_img, subcort_overlay, all_neg_img, all_pos_img,2.0,data_all_max,data_all_max,3)
Image.fromarray(img_all_LH.astype('uint8'), 'RGB').save(FIGURE_FILE_ALL_LH)
Image.fromarray(img_all_RH.astype('uint8'), 'RGB').save(FIGURE_FILE_ALL_RH)
Image.fromarray(img_all_axial.astype('uint8'), 'RGB').save(FIGURE_FILE_ALL_AXIAL)

#%% Combined image for figures
def combine_figures(img_LH,img_RH,img_axial):
    x,y,z = img_LH.shape
    img_new = np.empty((x,y*3,z))
    img_new[:,:y,:] = img_LH
    img_new[:,y:2*y,:] = img_axial
    img_new[:,2*y:,:] = np.flip(img_RH,axis=1)
    return img_new

img_YA_combined = combine_figures(img_YA_LH,img_YA_RH,img_YA_axial)
Image.fromarray(img_YA_combined.astype('uint8'),'RGB').save(FIGURE_FILE_YA_COMBINED)
img_OA_combined = combine_figures(img_OA_LH,img_OA_RH,img_OA_axial)
Image.fromarray(img_OA_combined.astype('uint8'),'RGB').save(FIGURE_FILE_OA_COMBINED)
img_all_combined = combine_figures(img_all_LH,img_all_RH,img_all_axial)
Image.fromarray(img_all_combined.astype('uint8'),'RGB').save(FIGURE_FILE_ALL_COMBINED)

#%%
a = np.outer(np.arange(0, 1, 0.01), np.ones(10))

plt.imshow(a, cmap=yellow_red_cmap, origin='lower')
plt.axis("off")

#%%
plt.imshow(a, cmap=green_blue_cmap, origin='lower')
plt.axis("off")