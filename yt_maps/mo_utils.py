#%run ~/Software/arepo-utils/startup.py
import copy
import os

import astropy.constants as ac
import astropy.units as u
import calcGrid
import h5py
import matplotlib.pyplot as plt
import numpy as np
from emis import emissivity
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class DataLoader():
    def __init__(self,fname,convert_to_proper=True,aexp=0.999,h=0.6774):
        self.fname = fname
        self.subfind_id = self.fname.split('/')[-1].split('.hdf5')[0]
        with h5py.File(fname, 'r') as f:
            gas = f['PartType0']
            self.rotpos = np.array(gas['RotatedCoordinates'])
            self.xe = np.array(gas['ElectronAbundance']) 
            self.int_energy = np.array(gas['InternalEnergy']) * (u.km/u.s)**2
            self.density = np.array(gas['Density']) * 1e10 *(u.Msun / u.kpc**3)
            self.met = np.array(gas['GFM_Metallicity']) #unitless
            self.mass = np.array(gas['Masses']) * 1e10 * u.Msun 
            self.volume = self.mass / self.density
            self.velocity = np.array(gas['RotatedVelocities']) * (u.km/u.s)
            self.gfm = np.array(gas['GFM_Metals'])
            self.ha_frac = self.gfm[:,0]
            self.n_frac = self.gfm[:,3]
            self.o_frac = self.gfm[:,4]
        
        if convert_to_proper:
            self.convert_comoving_to_proper(aexp=aexp,h=h) 
        
        self.n_H = (self.ha_frac * self.density / ac.m_p).to(1/u.cm**3)
        self.temperature = self.compute_temperatures() * u.K 
        
        self.e_halpha = 10**emissivity('H-alpha',self.n_H.value,self.met,self.temperature.value) * self.ha_frac 
        self.L_halpha = ((self.e_halpha*u.erg/u.s/u.cm**3) * self.volume).to(u.erg/u.s).value 
        #self.total_halpha_luminosity = np.sum(self.L_healpha)
        self.e_n2 = 10**emissivity('NII',self.n_H.value,self.met,self.temperature.value) * self.n_frac
        #self.L_n2 = ((e_n2*u.erg/u.s/u.cm**3) * self.volume).to(u.erg/u.s).value
        self.e_o3 = 10**emissivity('OIII-1',self.n_H.value,self.met,self.temperature.value) * self.o_frac
        #self.L_o3 = ((e_o3*u.erg/u.s/u.cm**3) * self.volume).to(u.erg/u.s).value
        
    
    def convert_comoving_to_proper(self,aexp=0.999,h=0.6774):
        """Convert comoving quantities to proper quantities"""
        self.mass = self.mass / h
        self.density = self.density * h**2 / aexp**3 
        self.volume = self.mass / self.density 
        self.rotpos = self.rotpos * aexp / h
        self.velocity = self.velocity / np.sqrt(aexp)

    def compute_temperatures(self,nh_thresh=0.13):
        mproton = ac.m_p
        kB = ac.k_B
        Xh = 0.76 
        gamma = 5/3. 
        mu = (4 / (1+3*Xh+4*Xh*self.xe)) * mproton 
        T = (gamma-1) * (self.int_energy/kB) * mu 
        
        sf_gas, = np.where(self.n_H.value>nh_thresh)
        T = T.to(u.K).value
        T[sf_gas] = 1000. 
        return T
    
    def make_projection_map(self,line,boxx,boxy,boxz,axes=[2,0],res=512,vel_center=0,vel_width=None):
        img_cent = [0,0,0]
        integration_axis = [i for i in [0,1,2] if i not in axes]
        if vel_width is not None:
            vel_cut, = np.where(np.abs(self.velocity.value[:,integration_axis[0]] - vel_center)>(vel_width/2.0))
            # set particles with velocities outside of the LOS range to have emissivities of 0.0.
            if line=='Halpha_SB':
                value_integrate = copy.deepcopy(self.e_halpha) # per cell emissivity
            elif line=='OIII_SB':
                value_integrate = copy.deepcopy(self.e_o3)
            elif line=='NII_SB': 
                value_integrate = copy.deepcopy(self.e_n2)
            value_integrate[vel_cut] = 0
        total_lum = np.sum(((value_integrate*u.erg/u.s/u.cm**3) * self.volume).to(u.erg/u.s).value)
        print(f"Total Luminosity of Cells: {total_lum} erg/s")
        #D_dummy = 10 * u.Mpc 
        #value_integrate = value_integrate / (4*np.pi*(D_dummy.to(u.cm).value)**2) #now per cell erg/s/cm2
        #d_per_pix = ((boxx*u.kpc)/res)
        #solid_angle_per_pix = (((d_per_pix/D_dummy)**2).to(u.Unit('')) * u.radian**2).to(u.arcsec**2).value
        #value_integrate = value_integrate / solid_angle_per_pix #now per cell erg/s/cm2/arcsec2


        proj = get_Aslice(self.rotpos,value_integrate, box = [boxx,boxy], center = img_cent, nx = res, ny = res, nz = 2*res, boxz = boxz, axes = axes, proj = True, numthreads=16)
        #sb_map = proj['grid']
        mapp = proj['grid'] * u.erg / u.s / u.cm**3
        # map is a sum along the line of sight --- need to multiply by the box length in cm 
        pixlength=(boxx * u.kpc) / res
        mapp = mapp * pixlength # should be the same as multiplying each value by dx 
        # we've summed along dx erg/s/cm^3 
        # so now we are in erg/s/cm^2 
        print(f'sum of map after multiplying by boxlength: {np.sum(mapp.to(u.erg/u.s/u.cm**2).value)}')
        d_per_pix = (boxx*u.kpc)/res # length of each pixel
        pixel_area = d_per_pix **2
        mapp = mapp * pixel_area # we now have erg/s luminosity in each pixel
        print(f"Map Total Luminosity erg/s before placement: {np.sum(mapp.to(u.erg/u.s).value)}")
        D_dummy = 10 * u.Mpc 
        solid_angle_per_pix = ((d_per_pix/D_dummy)**2).to(u.Unit('')) * u.radian**2 
        sb_map = mapp / solid_angle_per_pix
        sb_map = sb_map / (4*np.pi*D_dummy**2)
        sb_map = sb_map.to(u.erg/u.s/u.cm**2/u.arcsec**2).value
        
        return ProjMap(proj,sb_map,total_lum)

    def save_hdf5_stack(self,savedir,vels=[-250,0,250],width=390,res=512):
        fpath = f'{savedir}/{self.subfind_id}/{self.subfind_id}_maps.hdf5'
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        if os.path.exists(fpath):
            print(f'maps already made for {self.subfind_id}')
            return 
        if not os.path.exists(f'{savedir}/{self.subfind_id}'):
            os.mkdir(f'{savedir}/{self.subfind_id}')
        hf = h5py.File(fpath,'w')

        for line in ["Halpha_SB"]:
            group = hf.create_group(line)
            edge_group = group.create_group('edge-on')
            face_group = group.create_group('face-on')
            for i in vels:
                projmap = self.make_projection_map(line,
                                                   axes=[2,1],
                                                   boxx=400/0.667,
                                                   boxy=400/0.667,
                                                   boxz=800/0.667,
                                                   res=res,
                                                   vel_center=i,
                                                   vel_width=width)
                edge_group.create_dataset(f'{i}',data=projmap.map)
                edge_group.create_dataset(f'total_lum_{i}',data=projmap.total_lum)
                projmap2 = self.make_projection_map(line,
                                                   axes=[1,0],
                                                   boxx=400/0.667,
                                                   boxy=400/0.667,
                                                   boxz=800/0.667,
                                                   res=res,
                                                   vel_center=i,
                                                   vel_width=width)
                face_group.create_dataset(f'{i}',data=projmap2.map)
                face_group.create_dataset(f'total_lum_{i}',data=projmap2.total_lum)
        hf.close()

                



class ProjMap():
    def __init__(self,proj,mapp,total_lum=0,noise_floor=1e-24):
        self.proj = proj
        self.map = mapp
        self.noise_floor = noise_floor
        self.map[self.map<self.noise_floor] = self.noise_floor
        self.total_lum=total_lum
    def plot_map(self,vmin=-20.5,vmax=-18):
        if vmin <= np.log10(self.noise_floor):
            raise AssertionError('vmin for plot should not be less than the arbitrary imposed noise floor.')
        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.pcolormesh( self.proj['x'], self.proj['y'], np.log10(self.map),vmin=vmin,vmax=vmax, cmap = 'magma', rasterized = True )
        #im2 = ax[1].pcolormesh( maps.proj_xz['x'], maps.proj_xz['y'], np.log10(maps.map_xz),vmin=-20.5,vmax=-18, cmap = 'magma', rasterized = True)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('right', size='7%', pad='2%')
        plt.colorbar(im,cax)
        plt.tight_layout()
        return fig, ax 




    



def get_Aslice(pos,value, res=1024, center=False,center_list=None, axes=[0,1], proj_fact=0.5, box=False,boxsize=None, proj=False, nx=None, ny=None, nz=None, boxz=None, numthreads=1 ):
    if type( center ) == list:
        center = np.array( center )
    elif type( center ) != np.ndarray:
        center = center_list

    if type( box ) == list:
        box = np.array( box )
    elif type( box ) != np.ndarray:
        box = np.array( [boxsize,boxsize] )

    axis0 = axes[0]
    axis1 = axes[1]

    c = np.zeros( 3 )
    c[0] = center[axis0]
    c[1] = center[axis1]
    c[2] = center[3 - axis0 - axis1]


    px = np.abs( pos[:,axis0] - c[0] )
    py = np.abs( pos[:,axis1] - c[1] )
    pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

    if nz is None:
        nz = int(2*proj_fact*res)

    if proj:
        if boxz is not None:
            zdist = 0.5 * boxz
        else:
            if box.shape[0] == 3:
                zdist = proj_fact * box[2]
            else:
                zdist = proj_fact * box.max()
                boxz = 2.*zdist

    pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )

    print(f"Selected {len(pp)} particles.")

    if proj:
            print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist, box[0], box[1]) )
    if nx is None:
        nx = res
    if ny is None:
        ny = res

    posdata = pos[pp,:]
    valdata = value[pp].astype('float64')


    data = calcGrid.calcASlice( posdata, valdata, nx, ny, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads)
    slice = data[ "grid"]

    print("Total is ", slice.sum() * box.max() / res)


    data['x'] = np.arange( nx+1, dtype="float64" ) / nx * box[0] - .5 * box[0] + c[0]
    data['y'] = np.arange( ny+1, dtype="float64" ) / ny * box[1] - .5 * box[1] + c[1]
    

    return data

def load_and_plot(fname,orientation,line,vmin=-20.5,vmax=-18,noise_floor=1e-24):
    
    with h5py.File(fname,'r') as f:
        #low = np.array(f[line][orientation]['-250'])
        zero = np.array(f[line][orientation]['0'])
        #high = np.array(f[line][orientation]['250'])
    low[low<noise_floor] = noise_floor
    zero[zero<noise_floor] = noise_floor 
    high[high<noise_floor] = noise_floor

    fig, ax = plt.subplots(1,3,figsize=(34,10))
    im0 = ax[0].imshow(np.log10(low),origin='lower',cmap = 'magma', rasterized = True,vmin=vmin,vmax=vmax )
    im1 = ax[1].imshow(np.log10(zero),origin='lower',cmap = 'magma', rasterized = True,vmin=vmin,vmax=vmax )
    im2 = ax[2].imshow(np.log10(high),origin='lower',cmap = 'magma', rasterized = True,vmin=vmin,vmax=vmax )

    ax[0].text(0.98,0.98,r'$-$250 km s$^{-1}$',color='w',fontsize=30,transform=ax[0].transAxes,ha='right',va='top')
    ax[1].text(0.98,0.98,r'0 km s$^{-1}$',color='w',fontsize=30,transform=ax[1].transAxes,ha='right',va='top')
    ax[2].text(0.98,0.98,r'250 km s$^{-1}$',color='w',fontsize=30,transform=ax[2].transAxes,ha='right',va='top')
    ax_divider = make_axes_locatable(ax[2])
    cax = ax_divider.append_axes('right', size='7%', pad='2%')
    cbar = plt.colorbar(im2,cax)
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    ax[2].set_box_aspect(1)
    for i in ax:
        i.set_xticks([])
        i.set_yticks([])
    plt.subplots_adjust(wspace=0.05)
    return fig, ax