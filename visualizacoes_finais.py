# %%
# import libraries
import os, sys
from matplotlib_surface_plotting import plot_surf
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import hcp_utils as hcp
from scipy.stats import mannwhitneyu, wilcoxon
import seaborn as sns
import pandas as pd
N_REGIONS = 200
modelos = [ 'hard_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1',
            # 'SHARED_copy_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v3', 
            # 'SHARED_copy_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v4',
            # 'SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v2',
            'Single_MMSE_Score_v2', 'Single_PMAT_v2', 'Single_Gender_v1'
            ]
dlabelnii = nib.load('melodic_IC_ftb.dlabel.nii')
dlabelData= dlabelnii.get_fdata()

# %% [markdown]
# Funcoes

# %%
def criaStatsMap_v2(img, SCORES, nome, path, threshold = 0.001):
    data = img.get_fdata()
    vet = np.copy(data[0])
    for i in range(200):
        vet[vet == i] = SCORES[i]
    vetFinal = vet[None,:]
    img_map = nib.Cifti2Image(vetFinal, header=img.header)
    nib.save(img_map, path + '/vetScores_'+nome+'.nii')
    Xp = hcp.parcellate(vetFinal, hcp.ca_network)
    df = hcp.ranking(Xp[0], hcp.ca_network)
    df.to_csv(path + 'raking_'+nome+'.csv')
    return plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(vet, fill=0), bg_map=hcp.mesh.sulc)

# %% [markdown]
# Lê os escores dos testes dos modelos 

# %%
img = nib.load('melodic_IC_ftb.dlabel.nii')

for modelo in modelos:
    finalBrainCell = None
    for i in range(5):
        nome = 'test_' + modelo+ 'fold'+str(i)
        
        if modelo.startswith('Single'):
            path = '../BrainGNN_single/scores/'+nome
        else:
            path = 'scores/'+nome
        final = np.array([])
        for root, subdirectories, files in os.walk(path):
            for subdirectory in set(subdirectories):
                s1 = np.load(os.path.join(root,subdirectory,'s1.npy'))
                s2 = np.load(os.path.join(root,subdirectory,'s2.npy'))
                w1 = np.load(os.path.join(root,subdirectory,'w1.npy'))
                w2 = np.load(os.path.join(root,subdirectory,'w2.npy'))
                perm1 = np.load(os.path.join(root,subdirectory,'perm1.npy'))
                perm2 = np.load(os.path.join(root,subdirectory,'perm2.npy'))
                perm1 = perm1.reshape(s1.shape)
                perm2 = perm2.reshape(s2.shape)
                N_REGIONS = 200
                matrix_pesos = [0.0] * N_REGIONS * perm1.shape[0]
                for subject, s1_subj in zip(perm1, s1):
                    for index_perm, i in enumerate(subject):
                        if s1_subj[index_perm] > 0.2:
                            matrix_pesos[i] = s1_subj[index_perm]
                
                arr_2d = np.reshape(matrix_pesos, (perm1.shape[0], N_REGIONS))
                if len(final)==0:
                    final = arr_2d
                else:
                    final = np.concatenate((final, arr_2d))
        if finalBrainCell is None:
            finalBrainCell = final  
        else:
            finalBrainCell =  finalBrainCell + final

    with open(path+'/scores'+nome+'.npy', 'wb') as f:
        np.save(f, finalBrainCell)
    fig = plt.figure(figsize=(8,5))
    plt.xlabel('ROIs', fontsize=14)
    plt.ylabel('Subjects', fontsize=14)
    plt.imshow(finalBrainCell)
    plt.colorbar( orientation = 'vertical')
    # fig.savefig(path+'/' + nome+'_all_subjects_scores.png')
    importance = np.median(finalBrainCell, axis=0)
    with open(path+'/median_scores'+nome+'.npy', 'wb') as f:
        np.save(f, importance)
    SCORES = importance.reshape(1, -1)[0].astype('float32')
    criaStatsMap_v2(img, SCORES, nome, path)
    # SCORES[aceita_todas] = 1
    # SCORES[intersectionAllRej] = SCORES[intersectionAllRej] * 0
    # criaStatsMap_v2(img, SCORES, nome + 'aceita', path)
        

# %% [markdown]
# Teste wilcoxon

# %%
a = np.load('scores/test_hard_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1fold4/scorestest_hard_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1fold4.npy')
pmat_scores = np.load('../../GNN/BrainGNN_single/scores/test_Single_PMAT_v2fold4/scorestest_Single_PMAT_v2fold4.npy')
mmse_scores = np.load('../../GNN/BrainGNN_single/scores/test_Single_MMSE_Score_v2fold4/scorestest_Single_MMSE_Score_v2fold4.npy')
gender_scores = np.load('../../GNN/BrainGNN_single/scores/test_Single_Gender_v1fold4/scorestest_Single_Gender_v1fold4.npy')
# b = np.load('vetor_scorestest_HCP_MTL_reg_gender_pmat_mmse_v4.npy')

mean_mtl = np.load('scores/test_hard_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1fold4/median_scorestest_hard_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1fold4.npy')
mean_single_pmat = np.load('../../GNN/BrainGNN_single/scores/test_Single_PMAT_v2fold4/median_scorestest_Single_PMAT_v2fold4.npy')
mean_single_mmse = np.load('../../GNN/BrainGNN_single/scores/test_Single_MMSE_Score_v2fold4/median_scorestest_Single_MMSE_Score_v2fold4.npy')
mean_single_gender = np.load('../../GNN/BrainGNN_single/scores/test_Single_Gender_v1fold4/median_scorestest_Single_Gender_v1fold4.npy')

listRejeita = dict({'PMAT':list(), 'MMSE':list(), 'Gender':list()})
listAceita =  dict({'PMAT':list(), 'MMSE':list(), 'Gender':list()})
alternativaWill = 'greater' #two-side, greater, less
cols = pd.MultiIndex.from_product([['Single PMAT - MTL','Single MMSE Score - MTL', 'Single Gender - MTL'], ['W', 'p']])
dfResult = pd.DataFrame(columns=cols)
print(dfResult)
for region in range(N_REGIONS):
    dfResult.loc[region,:] = region
    dfResult['idRegion'] = region
    w, p = wilcoxon(a[:, region],pmat_scores[:, region], alternative=alternativaWill)
    dfResult.loc[region,'Single PMAT - MTL'] = (w, p)

    if p < 0.05:
        # print("Hipotese nula rejeitada, data are different in your case")
        listRejeita['PMAT'].append(region)
    else:
        # print("we have not enough evidences to reject H0")
        listAceita['PMAT'].append(region)
    
    w, p_ = wilcoxon(a[:, region],mmse_scores[:, region],  alternative=alternativaWill)
    dfResult.loc[region,'Single MMSE Score - MTL'] = (w, p_)

    if p_ < 0.05:
        # print("Hipotese nula rejeitada")
        listRejeita['MMSE'].append(region)
    else:
        # print("Hipotese nula aceita")
        listAceita['MMSE'].append(region)

    w, p_ = wilcoxon(a[:, region],gender_scores[:, region], alternative=alternativaWill)
    dfResult.loc[region,'Single Gender - MTL'] = (w, p_)

    if p_ < 0.05:
        # print("Hipotese nula rejeitada")
        listRejeita['Gender'].append(region)
    else:
        # print("Hipotese nula aceita")
        listAceita['Gender'].append(region)

print("Aceitam: PMAT {:%d} MMSE {:%d} Gender {:%d}" % (len(listAceita['PMAT']), len(listAceita['MMSE']),len(listAceita['Gender'])))
print("Rejeitam: PMAT {:%d} MMSE {:%d} Gender {:%d}" % (len(listRejeita['PMAT']), len(listRejeita['MMSE']), len(listRejeita['Gender'])))

dictTaks = {
    'PMAT' : {'label' : 'Inteligência Fluida', 'meanScore': mean_single_pmat, 'scoresTodosSujeitos': pmat_scores},
    'Gender' : {'label' : 'Gênero', 'meanScore': mean_single_gender, 'scoresTodosSujeitos': gender_scores},
    'MMSE' : {'label' : 'Escore Cognitivo', 'meanScore': mean_single_mmse, 'scoresTodosSujeitos': mmse_scores},
    'MTLv1':   {'label' :'MTL Hard', 'meanScore': mean_mtl, 'scoresTodosSujeitos': a},
}

arrayTasks = ['PMAT', 'Gender', 'MMSE']
colorMap = 'Set3'
for task in arrayTasks:
    print(task)
    SCORES = dictTaks[task]['meanScore']
    SCORES_MTL = dictTaks['MTLv1']['meanScore']
    vet = np.copy(dlabelData[0])
    zmap = np.zeros_like(dlabelData[0])
    zmapMTL = np.zeros_like(dlabelData[0])
    for i in listRejeita[task]:
        zmap[np.where(vet==i)[0]] = SCORES[i]
        zmapMTL[np.where(vet==i)[0]] = SCORES_MTL[i]
    # plotting.plot_surf_stat_map(hcp.mesh.inflated_right, hcp.right_cortex_data(zmapMTL, fill=0), hemi='right',
    #                                     title='Surface right hemisphere Média dos escores rejeita h0 Single ' + str(task), cmap=colorMap,
    #                                     bg_map=hcp.mesh.sulc_right,  output_file=task+'_right_surface.png')

    # plotting.plot_surf_stat_map(hcp.mesh.inflated_left, hcp.left_cortex_data(zmap, fill=0), hemi='left',
    #                                     title='Surface left hemisphere Média dos escores rejeita h0 Single '  + str(task), cmap=colorMap,
    #                                     bg_map=hcp.mesh.sulc_left, colorbar=False, output_file=task+'_left_surface.png')

    # plotting.plot_surf_stat_map(hcp.mesh.inflated_right, hcp.right_cortex_data(zmapMTL, fill=0), hemi='right',
    #                                     title='Surface right hemisphere Média dos escores rejeita h0 MTL', cmap=colorMap,
    #                                     bg_map=hcp.mesh.sulc_right, output_file=task+'_MTL_right_surface.png')

    # plotting.plot_surf_stat_map(hcp.mesh.inflated_left, hcp.left_cortex_data(zmapMTL, fill=0), hemi='left',
    #                                     title='Surface left hemisphere Média dos escores rejeita h0 MTL', cmap=colorMap,
    #                                     bg_map=hcp.mesh.sulc_left, colorbar=False, output_file=task+'_MTL_left_surface.png')

    vertices, faces=  hcp.mesh.inflated_left
    plot_surf(vertices, faces, hcp.left_cortex_data(zmap, fill=0), cmap=colorMap,rotate=[90,270],parcel=hcp.left_cortex_data(hcp.yeo7.map_all), parcel_cmap=hcp.ca_network.rgba,filled_parcels=False, label=True,colorbar=False,  filename=task+'_left_surface_medial.png')
    plot_surf(vertices, faces, hcp.left_cortex_data(zmapMTL, fill=0), cmap=colorMap,rotate=[90,270],parcel=hcp.left_cortex_data(hcp.yeo7.map_all), parcel_cmap=hcp.ca_network.rgba,filled_parcels=False,label=True,colorbar=False, filename=task+'_MTL_left_surface_medial.png')
    vertices, faces=  hcp.mesh.inflated_right
    # plot_surf(vertices, faces, hcp.right_cortex_data(zmapMTL, fill=0), cmap='bwr',rotate=[90,270],parcel=hcp.right_cortex_data(hcp.yeo7.map_all), parcel_cmap=hcp.yeo7.rgba,filled_parcels=True)
    plot_surf(vertices, faces, hcp.right_cortex_data(zmapMTL, fill=0), cmap=colorMap,rotate=[90,270],parcel=hcp.right_cortex_data(hcp.yeo7.map_all), parcel_cmap=hcp.ca_network.rgba,filled_parcels=False, label=True,colorbar=False, filename=task+'_MTL_right_surface_medial.png')
    plot_surf(vertices, faces, hcp.right_cortex_data(zmap, fill=0), cmap=colorMap,rotate=[90,270],parcel=hcp.right_cortex_data(hcp.yeo7.map_all), parcel_cmap=hcp.ca_network.rgba,filled_parcels=False, label=True,colorbar=False, filename=task+'_right_surface_medial.png')



# %%
dfResult.to_csv("wilcoxon_"+ alternativaWill+".csv")
dfResult.head()

# %%
dfResult = pd.read_csv("wilcoxon_"+ alternativaWill+".csv",header=[0,1])
dfResult.head()

# %%
intersection = set(listRejeita['Gender']).intersection(listRejeita['MMSE'])
intersectionAllRej = intersection.intersection(listRejeita['PMAT'])
intersectionAllRej = list(set(intersectionAllRej))
len(intersectionAllRej)

# %%
dfResult.iloc[intersectionAllRej].index

# %%
dfResult.iloc[132]['Single PMAT - MTL']['p']

# %%
arrayTasks = ['PMAT', 'Gender', 'MMSE']
colorMap = 'Set3'
for task in arrayTasks:
    print(task)
    SCORES = dictTaks[task]['meanScore']
    SCORES_MTL = dictTaks['MTLv1']['meanScore']
    vet = np.copy(dlabelData[0])
    zmap = np.zeros_like(dlabelData[0])
    zmapMTL = np.zeros_like(dlabelData[0])
    for i in dfResult.iloc[intersectionAllRej].index.to_list():
        zmap[np.where(vet==i)[0]] = SCORES[i]
        zmapMTL[np.where(vet==i)[0]] = SCORES_MTL[i]
    vertices, faces=  hcp.mesh.inflated_left
    plot_surf(vertices, faces, hcp.left_cortex_data(zmap, fill=0), cmap=colorMap,rotate=[90,270],colorbar=True,filename=task+'_rejeita_left.png')
    plot_surf(vertices, faces, hcp.left_cortex_data(zmapMTL, fill=0), cmap=colorMap,rotate=[90,270],colorbar=True,filename=task+'_MTL_rejeita_left.png')

    vertices, faces=  hcp.mesh.inflated_right
    plot_surf(vertices, faces, hcp.right_cortex_data(zmap, fill=0), cmap=colorMap,rotate=[90,270],colorbar=True, filename=task+'_rejeita_right.png')
    plot_surf(vertices, faces, hcp.right_cortex_data(zmapMTL, fill=0), cmap=colorMap,rotate=[90,270],colorbar=True, filename=task+'_MTL_rejeita_right.png')


# %% [markdown]
# # considerando só os escores dessas regiões que foram rejeitadas

# %%
img = nib.load('melodic_IC_ftb.dlabel.nii')

for modelo in modelos:
    finalBrainCell = None
    for i in range(5):
        nome = 'test_' + modelo+ 'fold'+str(i)
        
        if modelo.startswith('Single'):
            path = '../BrainGNN_single/scores/'+nome
        else:
            path = 'scores/'+nome
        final = np.array([])
        for root, subdirectories, files in os.walk(path):
            for subdirectory in set(subdirectories):
                s1 = np.load(os.path.join(root,subdirectory,'s1.npy'))
                s2 = np.load(os.path.join(root,subdirectory,'s2.npy'))
                w1 = np.load(os.path.join(root,subdirectory,'w1.npy'))
                w2 = np.load(os.path.join(root,subdirectory,'w2.npy'))
                perm1 = np.load(os.path.join(root,subdirectory,'perm1.npy'))
                perm2 = np.load(os.path.join(root,subdirectory,'perm2.npy'))
                perm1 = perm1.reshape(s1.shape)
                perm2 = perm2.reshape(s2.shape)
                N_REGIONS = 200
                matrix_pesos = [0.0] * N_REGIONS * perm1.shape[0]
                for subject, s1_subj in zip(perm1, s1):
                    for index_perm, i in enumerate(subject):
                        if s1_subj[index_perm] > 0.2:
                            matrix_pesos[i] = s1_subj[index_perm]
                
                arr_2d = np.reshape(matrix_pesos, (perm1.shape[0], N_REGIONS))
                if len(final)==0:
                    final = arr_2d
                else:
                    final = np.concatenate((final, arr_2d))
        if finalBrainCell is None:
            finalBrainCell = final  
        else:
            finalBrainCell =  finalBrainCell + final

    with open(path+'/scores'+nome+'.npy', 'wb') as f:
        np.save(f, finalBrainCell)
    fig = plt.figure(figsize=(8,5))
    plt.xlabel('ROIs', fontsize=14)
    plt.ylabel('Subjects', fontsize=14)
    plt.imshow(finalBrainCell)
    plt.colorbar( orientation = 'vertical')
    # fig.savefig(path+'/' + nome+'_all_subjects_scores.png')
    importance = np.median(final, axis=0)
    with open(path+'/median_scores'+nome+'.npy', 'wb') as f:
        np.save(f, importance)
    SCORES = importance.reshape(1, -1)[0].astype('float32')
    criaStatsMap_v2(img, SCORES, nome, path)
    # SCORES[aceita_todas] = 1
    # SCORES[intersectionAllRej] = SCORES[intersectionAllRej] * 0
    # criaStatsMap_v2(img, SCORES, nome + 'aceita', path)
        

# %%
dfResult.iloc[intersectionAllRej].to_csv("wilcoxon_"+ alternativaWill+"_intersectionAllRej.csv")

# %% [markdown]
# # Visualizar os escores in surface

# %%
plotting.view_surf(hcp.mesh.inflated_right, hcp.right_cortex_data(zmapMTL), bg_map=hcp.mesh.sulc_right, cmap='bwr')


