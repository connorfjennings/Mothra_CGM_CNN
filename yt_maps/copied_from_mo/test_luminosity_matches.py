import glob

import astropy.constants as ac
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

face_truth_luminosities = []
face_computed_luminosities = []
edge_truth_luminosities = []
edge_computed_luminosities = []


def compute_map_luminosity(mapgrid,boxwidth=(400/0.667)*u.kpc,pixres=1024):
    D_dummy = 10 * u.Mpc 
    arcsec = (((boxwidth/pixres) / D_dummy ).to(u.Unit(''))*u.radian).to(u.arcsec)
    mapgrid = mapgrid*u.erg/u.s/u.cm**2/u.arcsec**2 
    mapgrid = mapgrid * arcsec**2 
    mapgrid = mapgrid * (4*np.pi*D_dummy**2)
    mapgrid = mapgrid.to(u.erg/u.s)
    tot_lum = np.sum(mapgrid.value)
    return tot_lum


dirs = glob.glob('hrh3/*')

for i in dirs: 
    oid = i.split('/')[1]
    fname = i + f'/{oid}_maps.hdf5'
    try:
        with h5py.File(fname) as f:
            face_truth_luminosities.append(np.float64(f["Halpha_SB"]["face-on"]['total_lum']))
            edge_truth_luminosities.append(np.float64(f["Halpha_SB"]["edge-on"]['total_lum']))    
            mapp_edge = np.array(f['Halpha_SB']['edge-on']['0'])
            edge_computed_luminosities.append(compute_map_luminosity(mapp_edge))
            mapp_face = np.array(f['Halpha_SB']['face-on']['0'])
            face_computed_luminosities.append(compute_map_luminosity(mapp_face))

    except Exception as e: 
        print(e)
        continue





fig, ax = plt.subplots(figsize=(10,10))

ax.plot(face_truth_luminosities,face_computed_luminosities,'s')
ax.plot(edge_truth_luminosities,edge_computed_luminosities,'o')
xx = np.linspace(np.min(face_truth_luminosities),np.max(face_truth_luminosities),10)
yy = xx 
ax.plot(xx,yy,'k')
plt.show()