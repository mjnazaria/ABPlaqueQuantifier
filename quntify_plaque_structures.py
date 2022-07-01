
"""
Quantify plaque density across different mouse lines, age groups and brain structures 
and saves the values in the data_plaque_group.csv file.

inputs
----------
original image stack path
segmented image stack path
main data frame path
mouse lines, animal id, age, sex

output
-------
Saves data_plaque_group.csv file
"""


from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile as tiff
import pandas as pd
import nrrd
import seaborn as sns
from scipy import signal

path_df = 'data_plaque_group.csv'
path_img_seg = 'segmented_registered_image.tif'
path_img_og = 'registered_image.tif'
animal_id = '00'
mouse_line = '5xFAD'
age_group = '6 mo'
age = 175
sex = 'M'

img_stack_seg = tiff.imread(path_img_seg)
img_stack_og = tiff.imread(path_img_og)
img_stack_og = img_stack_og>1
df_main_group = pd.read_csv(path_df, index_col=0)

mcc = MouseConnectivityCache(resolution=25)
rsp = mcc.get_reference_space()
structures = rsp.remove_unassigned()
structures = pd.DataFrame(structures)
acronym_id_map = rsp.structure_tree.get_id_acronym_map()
id_acronym_map = {v:k for k,v in acronym_id_map.items()} 
structures_leaves = [id_acronym_map[x] for x in structures['id'] if len(rsp.structure_tree.child_ids([x])[0])==0]    
masks_arg = {}
for id in structures['id']:
    masks_arg[id] = np.nonzero(rsp.make_structure_mask([id], direct_only=False))

def get_leaves(id):
    descents = rsp.structure_tree.descendant_ids([id])[0]
    leaves = list(map(acronym_id_map.get, structures_leaves))
    return [des for des in descents if (des in leaves)]

def main():

    df_main = pd.DataFrame(columns = ['animal_id', 'mouse_line', 'sex', 'age_group', 'age', 'id', 'acronym', 
                                      'volume', 'volume_in', 'plaque_volume'])
    df_main['animal_id'] = animal_id
    df_main['mouse_line'] = mouse_line
    df_main['sex'] = sex
    df_main['age_group'] = age_group
    df_main['age'] = age
    df_main['id'] = structures['id']
    df_main['acronym'] = structures['acronym']
    df_main['volume'] = df_main['id'].map(lambda x: len(masks_arg[x][0]))
    df_main['volume_in'] = df_main['volume']
    df_main['plaque_volume'] = df_main['id'].map(lambda x: np.sum(img_stack_seg[masks_arg[x]]))
    
    # recalculate the volume of boundary structures
    boundary_struct_root  = ['CB', 'P', 'MY', 'OLF', 'fiber tracts']
    boundary_struct_leaves = []
    for acr in boundary_struct_root:
        boundary_struct_leaves.extend(get_leaves(acronym_id_map[acr]))
    boundary_sig = np.mean(img_stack_og.reshape(img_stack_og.shape[0], -1), 1)
    boundary_sig_slope = np.diff(boundary_sig)
    frames_lim = signal.find_peaks(np.abs(boundary_sig_slope), height=np.percentile(np.abs(boundary_sig_slope), 90))
    frames_lim = frames_lim[0][[0,-1]]
    frames_offborder = np.append(np.arange(frames_lim[0]+10), np.arange(frames_lim[1]-10,img_stack_og.shape[0])) 
    for id in boundary_struct_leaves:
        if len(np.intersect1d(frames_offborder, masks_arg[id][0]))>5:
            idx = df_main.query('id == @id').index[0]
            df_main.loc[idx, 'volume_in'] = np.sum(img_stack_og[masks_arg[id]])
            if df_main.loc[idx, 'volume_in'] < .8*df_main.loc[idx, 'volume']:
                df_main.loc[idx, 'volume_in'] = 0
    df_main['plaque_density'] = 100*df_main['plaque_volume']/(df_main['volume_in']+1e-9)
    df_main.loc[df_main['volume_in']<5, 'plaque_density'] = np.nan    
    df_main_group = pd.concat([df_main, df_main_group ], axis=0, ignore_index=True)
    df_main_group.to_csv(path_df)

if __name__ == '__main__':
    main()