import os, h5py, sys, numpy as np
import illustris_python as il
import yt
import yt.units as ytu
from emis import emissivity
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from plot_utils import quiver_from_components
from scipy.spatial.transform import Rotation as R

yt.enable_parallelism()

subid = int(sys.argv[1])
angle_number = int(sys.argv[2])
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

basefilename = f'TNG50_snap099_fofid{subid:06d}'
inpath = f'/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample/{basefilename}_gas_sphere_500kpc.hdf5'
idx_path = inpath.replace(".hdf5", ".ytindex")
ds = yt.load(inpath, index_filename=idx_path,smoothing_factor=2)

width_proper = (500.0, "kpc")                 # proper
center_comov = 0.5 * ds.domain_width          # code center (comoving)
center_prop  = (center_comov.to("kpc") * ds.scale_factor)
resolution = (256,256)

def _Ha_cloudy(field, data):
    nH = data["gas","H_number_density"].in_units("1/cm**3").astype("float64")
    T  = data["gas","temperature"].in_units("K").astype("float64")
    Z  = data["gas","metallicity"].to("dimensionless").astype("float64")                
    # your helper returns log10 emissivity
    log_eps = emissivity("H-alpha", nH.value, Z.value, T.value, redshift=float(data.ds.current_redshift))
    eps = np.power(10.0, log_eps, dtype=np.float64) * data['gas','H_fraction'] * ytu.erg/ytu.s/ytu.cm**3
    return eps
ds.add_field(
    name=("gas", "Halpha_emissivity"),
    function=_Ha_cloudy,
    sampling_type="particle",
    units="erg/s/cm**3"
)
def _Ha_brightness(field, data):
    return data[("gas","Halpha_emissivity")] / (4*np.pi) / ytu.sr
ds.add_field(
    name=("gas", "Halpha_brightness"),
    function=_Ha_brightness,
    sampling_type="particle",
    units="erg/s/cm**3/arcsec**2"
)

def _Ha_pow(field, data):
    return data[("gas","Halpha_emissivity")] * data[("gas","cell_volume")]
ds.add_field(
    name=("gas", "Halpha_power"),
    function=_Ha_pow,
    sampling_type="particle",
    units="erg/s"
)

def _mask(lo, hi):
    def _f(field, data):      
        dv = data[("gas","velocity_los")].to_value("km/s")
        return ((dv >= lo) & (dv < hi)).astype("float64")  # 1 inside, 0 outside
    ds.add_field(
        name=("gas", f"mask_{lo:.0f}_{hi:.0f}"),
        function=_f,
        units="",
        sampling_type="particle"
    )
    def _g(field, data):
        return data[("gas","Halpha_emissivity")] * data[("gas",f"mask_{lo:.0f}_{hi:.0f}")]
    ds.add_field(
        name=("gas", f"Halpha_emissivity_{lo:.0f}_{hi:.0f}"),
        function=_g,
        units="erg/s/cm**3",
        sampling_type="particle"
    )
    def _h(field, data):
        return data[("gas","Halpha_brightness")] * data[("gas",f"mask_{lo:.0f}_{hi:.0f}")]
    ds.add_field(
        name=("gas", f"Halpha_brightness_{lo:.0f}_{hi:.0f}"),
        function=_h,
        units="erg/s/cm**3/arcsec**2",
        sampling_type="particle"
    )

    return _f, _g

_mask(-300, -100); _mask(-100, 100); _mask(100, 300)

def _vel_u(field,data):
    ux,uy,uz = u_hat[0],u_hat[1],u_hat[2]
    vx = data['gas','relative_velocity_x']
    vy = data['gas','relative_velocity_y']
    vz = data['gas','relative_velocity_z']
    return vx*ux + vy*uy + vz*uz
ds.add_field(
    name=("gas", f"velocity_u"),
    function=_vel_u,
    units="km/s",
    sampling_type="particle"
)
def _momentum_density_u(field,data):
    return data['gas','velocity_u']*data['gas','density']
ds.add_field(name=('gas','momentum_density_u'),function=_momentum_density_u,units="km*g/cm**3/s",sampling_type="particle")

def _vel_v(field,data):
    ux,uy,uz = v_hat[0],v_hat[1],v_hat[2]
    vx = data['gas','relative_velocity_x']
    vy = data['gas','relative_velocity_y']
    vz = data['gas','relative_velocity_z']
    return vx*ux + vy*uy + vz*uz
ds.add_field(
    name=("gas", f"velocity_v"),
    function=_vel_v,
    units="km/s",
    sampling_type="particle"
)
def _momentum_density_v(field,data):
    return data['gas','velocity_v']*data['gas','density']
ds.add_field(name=('gas','momentum_density_v'),function=_momentum_density_v,units="km*g/cm**3/s",sampling_type="particle")

def _vel_w(field,data):
    ux,uy,uz = w_hat[0],w_hat[1],w_hat[2]
    vx = data['gas','relative_velocity_x']
    vy = data['gas','relative_velocity_y']
    vz = data['gas','relative_velocity_z']
    return vx*ux + vy*uy + vz*uz
ds.add_field(
    name=("gas", f"velocity_w"),
    function=_vel_w,
    units="km/s",
    sampling_type="particle"
)
def _momentum_density_w(field,data):
    return data['gas','velocity_w']*data['gas','density']
ds.add_field(name=('gas','momentum_density_w'),function=_momentum_density_w,units="km*g/cm**3/s",sampling_type="particle")



sp_small = ds.sphere(center_prop, (50.0, "kpc"))
bv = sp_small.quantities.bulk_velocity()
sp = ds.sphere(center_prop, (500.0, "kpc"))
sp.set_field_parameter("bulk_velocity", bv)

#save velocity maps
outdir = '/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/velocity/'

field = ("gas","velocity_u")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    weight_field=("gas","density"),
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))


field = ("gas","velocity_v")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    weight_field=("gas","density"),
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))

field = ("gas","velocity_w")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    weight_field=("gas","density"),
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))

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
    
#save momentum maps
outdir = '/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/momentum/'

field = ("gas","momentum_density_u")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))

field = ("gas","momentum_density_v")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))

field = ("gas","momentum_density_w")
p = yt.OffAxisProjectionPlot(
    ds, w_hat, field,
    center=center_prop, width=width_proper,
    north_vector=v_hat,
    buff_size=resolution,
    data_source=sp
)
img = p.frb[field]
outfile = f'{basefilename}_view{angle_number:02d}_{field[1]}.npy'
outpath = outdir+outfile
np.save(outpath,img.to_value(img.units).astype(np.float32))
