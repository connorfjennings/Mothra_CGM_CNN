import time
import requests 
import os
import h5py 
import numpy as np 

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"b1defa969bc029b45620e0592b90c93c"}

def get(path,subid, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        if not os.path.exists(f'new_haloes/{subid}'):
            os.mkdir(f'new_haloes/{subid}')
        filename = 'new_haloes/'+f'{subid}/'+r.headers['content-disposition'].split("filename=")[1]
        if os.path.exists(filename):
            return filename
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r


def download_subhalo_box(subid):
    url = baseUrl+f'TNG50-1/snapshots/99/subhalos/{subid}/mwm31s_cutouts/snap_99.hdf5'
    cutout_fname = get(url,subid)
    return cutout_fname

with h5py.File('mw_tng_sample_catalog.hdf5') as f:
    subfind_ids = np.array(f['SubfindID'])

all_downloaded=False 
index = 0
while not all_downloaded:
    if index == len(subfind_ids):
        all_downloaded=True
        break
    elif os.path.exists(f'new_haloes/{subfind_ids[index]}/{subfind_ids[index]}.hdf5'):
        print(f'already downloaded {subfind_ids[index]}')
        index+=1
        continue
    try:
        print(f'Starting on box: {subfind_ids[index]}')
        
        cutout_fname = download_subhalo_box(subfind_ids[index])
        assert os.path.exists(cutout_fname)
        index+=1
        print(f'Box {subfind_ids[index]} downloaded successfully. ({index}/{len(subfind_ids)})')
    except:
        time.sleep(60)
