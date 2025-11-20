import os, h5py, sys, numpy as np
import illustris_python as il
import yt
import yt.units as ytu
from emis import emissivity
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from plot_utils import quiver_from_components
from yt_field_utils import *
from scipy.spatial.transform import Rotation as R

yt.enable_parallelism()

subid = int(sys.argv[1])
print(subid)
angle_number = int(sys.argv[2])
print(angle_number)
particle_filter = sys.argv[3]
print(particle_filter)
theta_list = [['x',0,'y',0],
              ['x',0,'y',30],
              ['x',0,'y',60],
              ['x',0,'y',90],
              ['x',30,'y',0],
              ['x',60,'y',0],
              ['x',90,'y',0],
              ['x',90,'z',30],
              ['x',90,'z',60],
              ['x',45,'z',45]]
theta = theta_list[angle_number]
r1 = R.from_euler(theta[0], theta[1],degrees=True)
r2 = R.from_euler(theta[2], theta[3],degrees=True)
z_hat = np.array([0,0,1])
y_hat = np.array([0,1,0])
w_hat = r2.apply(r1.apply(z_hat))
v_hat = r2.apply(r1.apply(y_hat))
w_hat /= np.sqrt(np.dot(w_hat,w_hat))
v_hat /= np.sqrt(np.dot(v_hat,v_hat))
u_hat = np.cross(v_hat,w_hat)

basefilename = f'TNG50_snap099_subid{subid:06d}'
inpath = f'/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample/{basefilename}_gas_sphere_500kpc.hdf5'
idx_path = inpath.replace(".hdf5", ".ytindex")
ds = yt.load(inpath, index_filename=idx_path,smoothing_factor=2)

width_proper = (500.0, "kpc")                 # proper
center_comov = 0.5 * ds.domain_width          # code center (comoving)
center_prop  = (center_comov.to("kpc") * ds.scale_factor)
resolution = (256,256)

add_emission_fields(ds,line="Halpha")

vel_bounds = [(-300, -100),(-100, 100),(100, 300)]
for vb in vel_bounds:
    lo,hi = vb
    add_los_mask_fields(ds,lo, hi)
    add_los_mask_emission_fields(ds,lo,hi,line='Halpha')

add_spherical_rv(ds)

shell_midpoints = np.arange(20,205,5)
shell_names = []
delta_r = 5
for i in range(len(shell_midpoints)):
    rmid = int(shell_midpoints[i])
    add_shell_filter(ds,rmid,delta_r = delta_r)
    
    shell_names.append(f"shell_{rmid}")

sp_small = ds.sphere(center_prop, (50.0, "kpc"))
bv = sp_small.quantities.bulk_velocity()
sp = ds.sphere(center_prop, (500.0, "kpc"))
sp.set_field_parameter("bulk_velocity", bv)

add_velocity_projection_fields(ds,u_hat,v_hat,w_hat,cold=False,emission=False)

add_emission_filter(ds,minL=5e-43,line='Halpha')
add_coldgas_filter(ds,maxT=10**4.5)

outdir = '/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/Halpha5e-43velocity/'

for field in [(particle_filter,"velocity_u"),(particle_filter,"velocity_v"),(particle_filter,"velocity_w")]:
    p = yt.OffAxisProjectionPlot(
        ds, w_hat, field,
        center=center_prop, width=width_proper,
        north_vector=v_hat,
        weight_field=(particle_filter,"density"),
        buff_size=resolution,
        data_source=sp
    )
    img = p.frb[field]
    outfile = f'{basefilename}_view{angle_number:02d}_emitting_{field[1]}.npy'
    outpath = outdir+outfile
    np.save(outpath,img.to_value(img.units).astype(np.float32))

#radial summary
if angle_number == 0:
    outdir = '/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/1D_kinematics/'
    
    mass_flow_array = []
    mass_array = []
    for sn in shell_names:
        mask = joint_mask(sp,sn,particle_filter)
        vr = sp['gas','vel_sph_rad'][mask].in_units('km/s')
        m = sp['gas','mass'][mask].in_units('Msun')
        mass_flow = np.sum(vr * m) / (delta_r*ytu.pc)
        mass_flow_array.append(mass_flow.in_units('Msun/yr').value)
        mass_array.append(m.sum().in_units('Msun').value)
    mass_flow_array = np.array(mass_flow_array)
    mass_array = np.array(mass_array)
    kinematics_1D = np.stack([mass_array,mass_flow_array],axis=-1)
    outfile = f'{basefilename}_emitting_1D_kinematics.npy'
    np.save(outpath,kinematics_1D.astype(np.float32))

#save Halpha brightness maps
outdir = '/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/Halpha/'

field_list = [("gas","Halpha_brightness"),
              ("gas","Halpha_brightness_-300_-100"),
              ("gas","Halpha_brightness_-100_100"),
              ("gas","Halpha_brightness_100_300")]
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field_list,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    method="integrate",
    buff_size=resolution,
    data_source=sp
)

for field in field_list:
    img = p.frb[field]
    outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
    outpath = outdir+outfile
    np.save(outpath,img.to_value(img.units).astype(np.float32))





