
"""
Creates a series of bar plots for plaque density across different mouse lines, age groups and brain structures 
from values in the data_plaque_group.csv file.
"""

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path_df = 'data_plaque_group.csv'

df_main_group = pd.read_csv(path_df, index_col=0)
mcc = MouseConnectivityCache(resolution=25)
rsp = mcc.get_reference_space()
structures = rsp.remove_unassigned()
structures = pd.DataFrame(structures)
acronym_id_map = rsp.structure_tree.get_id_acronym_map() # Dictionary returns IDs given acronyms
id_acronym_map = {v:k for k,v in acronym_id_map.items()} # Flip key:value pairs to get dictionary for acronyms given IDs

def get_substructure(id):
    descents = rsp.structure_tree.descendants([id])[0]
    descents_clean = []
    for des in descents:
        if not any(char.isdigit() for char in des['name']):
            if len(rsp.structure_tree.child_ids([des['id']])[0])==0 and 'layer' not in des['name'].lower():
                descents_clean.append(des['acronym'])
            elif len(rsp.structure_tree.child_ids([des['id']])[0]) and 'layer' in rsp.structure_tree.children([des['id']])[0][0]['name'].lower():
                descents_clean.append(des['acronym'])
    return descents_clean

def roi_data_filter(rois):
    df_filt = df_main_group[df_main_group['acronym'].isin(rois)]
    return df_filt

def main():
    roi_course = pd.DataFrame(rsp.structure_tree.get_structures_by_set_id([2]))['acronym'].values
    roi_course = np.append(roi_course, 'fiber tracts')
    roi_isoctx = get_substructure(315)
    roi_hpf = get_substructure(1089)
    roi_olf = list(map(id_acronym_map.get, rsp.structure_tree.child_ids([698])[0]))
    roi_ctxsp = list(map(id_acronym_map.get, rsp.structure_tree.child_ids([703])[0]))
    ctx_layers = structures[structures['name'].map(lambda x: any(num in x for num in '123456'))]
    roi_ctx_layers = ctx_layers['acronym'][ctx_layers['structure_id_path'].map(lambda x: 315 in x)].values

    plt.figure(figsize=((8,4)))
    sns.barplot(x='age_group', hue='mouse_line', y='plaque_density', data=roi_data_filter(['root']), capsize=.2)
    plt.xlabel('')
    plt.ylabel('Plaque density (%)')

    plt.figure(figsize=((8,4)))
    sns.barplot(x='acronym', y='plaque_density', data=roi_data_filter(roi_course), capsize=.2)
    plt.xlabel('')
    plt.ylabel('Plaque density (%)')
    
    plt.figure()
    plt.subplot(121)
    sns.barplot(y='acronym', x='plaque_density', data=roi_data_filter(roi_isoctx), capsize=.2, orient='h')
    plt.ylabel('')
    plt.xlabel('Plaque density (%)')
    plt.title('Isocortex')
    plt.subplot(322)
    sns.barplot(y='acronym', x='plaque_density', data=roi_data_filter(roi_hpf), capsize=.2, orient='h')
    plt.gca().set(xlabel='', ylabel='')
    plt.title('Hippocampal Formation')   
    plt.subplot(324)
    sns.barplot(y='acronym', x='plaque_density', data=roi_data_filter(roi_olf), capsize=.2, orient='h')
    plt.gca().set(xlabel='', ylabel='')
    plt.title('Olfactory Areas')   
    plt.subplot(326)
    sns.barplot(y='acronym', x='plaque_density', data=roi_data_filter(roi_ctxsp), capsize=.2, orient='h')
    plt.ylabel('')
    plt.xlabel('Plaque density (%)')
    plt.title('Cortical Subplate')
    
    # cortical layers plaque density
    df_sub = df_main_group[df_main_group['acronym'].isin(roi_ctx_layers)]
    df_sub['layer'] = ''
    for num in ['1', '2/3', '4', '5', '6a', '6b']:
        df_sub.loc[df_sub['acronym'].str.contains(num), 'layer'] = num    
    df_sub = df_sub.groupby(['animal_id', 'layer']).agg({'volume': ['sum'], 'plaque_volume': ['sum']})
    df_sub = df_sub.droplevel(1, axis=1)
    df_sub = df_sub.droplevel(0, axis=0)
    df_sub['plaque_density'] = 100*df_sub['plaque_volume']/df_sub['volume']    
    sns.barplot(y=df_sub.index, x='plaque_density', data=df_sub, capsize=.2, orient='h')
    plt.ylabel('Layer')
    plt.xlabel('Plaque density (%)')

if __name__ == '__main__':
    main()
