import os, h5py, numpy as np
import illustris_python as il
import yt
import yt.units as ytu
from emis import emissivity
import astropy.units as u
import astropy.constants as c

def joint_mask(sp,filter1,filter2):
    pid_all = sp['gas','index']
    pid_1 = sp[filter1,'index']
    pid_2 = sp[filter2,'index']
    pid_both = np.intersect1d(pid_1, pid_2, assume_unique=False)
    mask_both = np.in1d(pid_all, pid_both, assume_unique=False)
    return mask_both


def add_emission_fields(ds,line="Halpha"):
    def _emission_cloudy(field, data):
        nH = data["gas","H_number_density"].in_units("1/cm**3").astype("float64")
        T  = data["gas","temperature"].in_units("K").astype("float64")
        T = np.where(T.value<1e4,1e4,T.value) #TNG50 has a different perscription for T<1e4, clip
        Z  = data["gas","metallicity"].to("dimensionless").astype("float64")

        if line=='Halpha':
            log_eps = emissivity("H-alpha", nH.value, Z.value, T, redshift=float(data.ds.current_redshift))
            eps = np.power(10.0, log_eps, dtype=np.float64) * data['gas','H_fraction'] * ytu.erg/ytu.s/ytu.cm**3
        elif line=='N2':
            log_eps = emissivity('NII', nH.value, Z.value, T, redshift=float(data.ds.current_redshift))
            eps = np.power(10.0, log_eps, dtype=np.float64) * data['gas','N_fraction'] * ytu.erg/ytu.s/ytu.cm**3
        elif line=='O3':
            log_eps = emissivity('OIII-1', nH.value, Z.value, T, redshift=float(data.ds.current_redshift))
            eps = np.power(10.0, log_eps, dtype=np.float64) * data['gas','O_fraction'] * ytu.erg/ytu.s/ytu.cm**3
        return eps
    ds.add_field(
        name=("gas", f"{line}_emissivity"),
        function=_emission_cloudy,
        sampling_type="particle",
        units="erg/s/cm**3"
    )
    def _brightness(field, data):
        return data[("gas",f"{line}_emissivity")] / (4*np.pi) / ytu.sr
    ds.add_field(
        name=("gas", f"{line}_brightness"),
        function=_brightness,
        sampling_type="particle",
        units="erg/s/cm**3/arcsec**2"
    )
    
    def _pow(field, data):
        return data[("gas",f"{line}_emissivity")] * data[("gas","cell_volume")]
    ds.add_field(
        name=("gas", f"{line}_power"),
        function=_pow,
        sampling_type="particle",
        units="erg/s"
    )

def add_coldgas_weight_fields(ds,maxT=2e4):
    #these fields are used as weights/cuts
    def _coldgas(field,data):
        T  = data["gas","temperature"].to_value("K")
        return (T < maxT).astype("float64")
    ds.add_field(
        name=("gas", "cold"),
        function=_coldgas,
        units="",
        sampling_type="particle"
    )
    def _cold_density(field,data):
        return data[('gas','density')]*data[("gas", "cold")]
    ds.add_field(
        name=("gas", "cold_density"),
        function=_cold_density,
        units="g/cm**3",
        sampling_type="particle"
    )
    
def add_emission_weight_fields(ds,minL=1e-43,line='Halpha'):
    def _emittinggas(field,data):
        L  = data["gas",f"{line}_brightness"].to_value("erg/s/cm**3/arcsec**2")
        return (L > minL).astype("float64")
    ds.add_field(
        name=("gas", "emitting"),
        function=_emittinggas,
        units="",
        sampling_type="particle"
    )
    def _emitting_density(field,data): #this is used only as a weight
        return data[('gas','density')]*data[("gas", "Halpha_emissivity")]
    ds.add_field(
        name=("gas", "emitting_density"),
        function=_emitting_density,
        units="g/cm**3",
        sampling_type="particle"
    )

def add_emission_filter(ds,minL=1e-43,line='Halpha'):
    def _emittinggas(field,data):
        L  = data["gas",f"{line}_brightness"].to_value("erg/s/cm**3/arcsec**2")
        return (L > minL)
    yt.add_particle_filter(
        f'{line}_emitting_gas', 
        function=_emittinggas,
        filtered_type="gas",               # change to your particle type
        requires=[f"{line}_brightness"]
    )
    ds.add_particle_filter(f'{line}_emitting_gas')

def add_coldgas_filter(ds,maxT=10**(4.5)):
    def _coldgas(field,data):
        T  = data["gas","temperature"].to_value("K")
        return (T < maxT)
    yt.add_particle_filter(
        'cold_gas', 
        function=_coldgas,
        filtered_type="gas",               # change to your particle type
        requires=["temperature"]
    )
    ds.add_particle_filter('cold_gas')

def add_los_mask_fields(ds,lo, hi):
    def _f(field, data):      
        dv = data[("gas","velocity_los")].to_value("km/s")
        return ((dv >= lo) & (dv < hi)).astype("float64")  # 1 inside, 0 outside
    ds.add_field(
        name=("gas", f"mask_{lo:.0f}_{hi:.0f}"),
        function=_f,
        units="",
        sampling_type="particle"
    )

def add_los_mask_emission_fields(ds,lo,hi,line='Halpha'):
    def _g(field, data):
        return data[("gas",f"{line}_emissivity")] * data[("gas",f"mask_{lo:.0f}_{hi:.0f}")]
    ds.add_field(
        name=("gas", f"{line}_emissivity_{lo:.0f}_{hi:.0f}"),
        function=_g,
        units="erg/s/cm**3",
        sampling_type="particle"
    )
    def _h(field, data):
        return data[("gas",f"{line}_brightness")] * data[("gas",f"mask_{lo:.0f}_{hi:.0f}")]
    ds.add_field(
        name=("gas", f"{line}_brightness_{lo:.0f}_{hi:.0f}"),
        function=_h,
        units="erg/s/cm**3/arcsec**2",
        sampling_type="particle"
    )

#_mask(-300, -100); _mask(-100, 100); _mask(100, 300)

    
def add_velocity_projection_fields(ds,u_hat,v_hat,w_hat,cold=False,emission=False):
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
    if cold:
        def _vel_u_cold(field,data):
            return data['gas','velocity_u']*data['gas','cold']
        ds.add_field(name=('gas','cold_velocity_u'),function=_vel_u_cold,units="km/s",sampling_type="particle")
        def _vel_v_cold(field,data):
            return data['gas','velocity_v']*data['gas','cold']
        ds.add_field(name=('gas','cold_velocity_v'),function=_vel_v_cold,units="km/s",sampling_type="particle")
        def _vel_w_cold(field,data):
            return data['gas','velocity_w']*data['gas','cold']
        ds.add_field(name=('gas','cold_velocity_w'),function=_vel_w_cold,units="km/s",sampling_type="particle")
    if emission:
        def _vel_u_emit(field,data):
            return data['gas','velocity_u']*data['gas','emitting']
        ds.add_field(name=('gas','emitting_velocity_u'),function=_vel_u_emit,units="km/s",sampling_type="particle")
        def _vel_v_emit(field,data):
            return data['gas','velocity_v']*data['gas','emitting']
        ds.add_field(name=('gas','emitting_velocity_v'),function=_vel_v_emit,units="km/s",sampling_type="particle")
        def _vel_w_emit(field,data):
            return data['gas','velocity_w']*data['gas','emitting']
        ds.add_field(name=('gas','emitting_velocity_w'),function=_vel_w_emit,units="km/s",sampling_type="particle")
        
def add_spherical_rv(ds):
    def _vel_sph_rad(field, data):
        
        # define the side length of the simulation domain in kpc
        box_side_length = (data.ds.domain_width[0].in_units('kpc'))/2.
        x     = data['gas', 'x'].in_units("kpc") - box_side_length
        y     = data['gas', 'y'].in_units("kpc") - box_side_length
        z     = data['gas', 'z'].in_units("kpc") - box_side_length
        
        r     = np.sqrt(x**2 + y**2 + z**2)
        
        theta = np.arctan2(y, x)
        phi   = np.arccos(z/r)
    
        vel_sph_rad = (data['gas', 'velocity_x'].in_units("km/s") * np.cos(theta) * np.sin(phi) +
                       data['gas', 'velocity_y'].in_units("km/s") * np.sin(theta) * np.sin(phi) +
                       data['gas', 'velocity_z'].in_units("km/s") * np.cos(phi))
    
        return (vel_sph_rad)
    
    
    # spherical radius
    def _spherical_r(field, data):
        
        # define the side length of the simulation domain in kpc
        box_side_length = (data.ds.domain_width[0].in_units('kpc'))/2.
        spherical_r   = np.sqrt((data['gas', 'x'].in_units("kpc") - box_side_length)**2 + 
                                (data['gas', 'y'].in_units("kpc") - box_side_length)**2 + 
                                (data['gas', 'z'].in_units("kpc") - box_side_length)**2)
        return (spherical_r)
    
    
    ds.add_field(("gas", "vel_sph_rad"), function=_vel_sph_rad, units="km/s", sampling_type="particle")
    ds.add_field(("gas", "spherical_r"), function=_spherical_r, units="kpc", sampling_type="particle")

def add_shell_filter(ds,rmid,delta_r=5):
    rmin,rmax = rmid-delta_r/2, rmid+delta_r/2
    
    def shell_filter(pfilter, data):
        # choose your particle family; for TNG gas is usually "PartType0"
        p = pfilter.filtered_type                # e.g., "PartType0"
        # Get particle positions (kpc)
        r = data[p, "spherical_r"].to("kpc")   # shape (N,3)
        return (r >= rmin) & (r < rmax)
    shell_name = f"shell_{rmid}"
    yt.add_particle_filter(
        shell_name, 
        function=shell_filter,
        filtered_type="gas",               # change to your particle type
        requires=["spherical_r"]
    )
    ds.add_particle_filter(shell_name)