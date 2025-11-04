
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils import DataLoader, ProjMap, load_and_plot

with h5py.File('mw_tng_sample_catalog.hdf5') as f:
    subfind_ids = np.array(f['SubfindID'])

use_ids = list(subfind_ids)
use_ids = ['342447']
for n,i in enumerate(use_ids):
    #print(f'On subfind id: {i} ({n}/{len(subfind_ids[:1])})')
    print(f'      loading data for {i}  | N: {n}/{len(use_ids)}')
    if os.path.exists(f'hrh3/{i}/{i}_Halpha_Lum-edge.pdf'):
        print(f'{subfind_ids} done, continuing...')
        continue
    try:
        d = DataLoader(f'new_haloes/{i}/{i}.hdf5')
        print('      saving hdf5 stack....')
        d.save_hdf5_stack(savedir='hrh3',res=1024,vels=[-250,250])
        #print('      making plots...')
        # fig, ax = load_and_plot(i,'edge-on','Halpha_Lum')
        # fig.savefig(f'higher_res_haloes/{i}/{i}_Halpha_Lum-edge.png')
        # fig.savefig(f'higher_res_haloes/{i}/{i}_Halpha_Lum-edge.pdf')
        # plt.close(fig)
        # fig, ax = load_and_plot(i,'face-on','Halpha_Lum')
        # fig.savefig(f'higher_res_haloes/{i}/{i}_Halpha_Lum-face.png')
        # fig.savefig(f'higher_res_haloes/{i}/{i}_Halpha_Lum-face.pdf')
        # plt.close(fig)
        # fig, ax = load_and_plot(i,'edge-on','OIII_Lum')
        # fig.savefig(f'new_haloes/{i}/{i}_OIII_Lum-edge.png')
        # fig.savefig(f'new_haloes/{i}/{i}_OIII_Lum-edge.pdf')
        # plt.close(fig)
        # fig, ax = load_and_plot(i,'face-on','OIII_Lum')
        # fig.savefig(f'new_haloes/{i}/{i}_OIII_Lum-face.png')
        # fig.savefig(f'new_haloes/{i}/{i}_OIII_Lum-face.pdf')
        # plt.close(fig)
        # fig, ax = load_and_plot(i,'edge-on','NII_Lum')
        # fig.savefig(f'new_haloes/{i}/{i}_NII_Lum-edge.png')
        # fig.savefig(f'new_haloes/{i}/{i}_NII_Lum-edge.pdf')
        # plt.close(fig)
        # fig, ax = load_and_plot(i,'face-on','NII_Lum')
        # fig.savefig(f'new_haloes/{i}/{i}_NII_Lum-face.png')
        # fig.savefig(f'new_haloes/{i}/{i}_NII_Lum-face.pdf')
        # plt.close(fig)
        print(f'      Done {i}.')
    except Exception as e:
        print(f'Could not do subfind {i}: {e}')
        continue

