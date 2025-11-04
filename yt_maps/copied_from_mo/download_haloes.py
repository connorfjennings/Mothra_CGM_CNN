import requests 

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"b1defa969bc029b45620e0592b90c93c"}



def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r


def download_subhalo_box(subid):
    url = baseUrl+f'TNG50-1/snapshots/99/subhalos/{subid}/mwm31s_cutouts/snap_99.hdf5'
    cutout = get(url)




    