import h5py
from scipy.ndimage import map_coordinates
import numpy as np
import collections

basePath = '/scratch11/ip259/CGM-MAPS/' ## basePath to cloudy tables, set accordingly.

lineAbbreviations = {'Lyman-alpha' : 'H  1 1215.67A',
                          'Lyman-beta'  : 'H  1 1025.72A',
                          'MgII'        : 'Blnd 2798.00A', # 2796+2803A together
                          'H-alpha'     : 'H  1 6562.81A',
                          'Halpha'     : 'H  1 6562.81A',
                          'H-beta'      : 'H  1 4861.33A',
                          '[OII]3729'   : 'O  2 3728.81A',
                          'OVII'        : 'O  7 22.1012A',
                          'OVIII'       : 'O  8 18.9709A',
                          'CVI'         : 'C  6 33.7372A',
                          'NVII'        : 'N  7 24.7807A',
                          'OIII-1'        : 'O  3 5006.84A',
                          'OIII'      : 'O  3 4958.91A',
                          'NII'         : 'N  2 6583.45A'
                    }

# proposed emission lines to record:
lineList = """
#1259    H  1 911.753A      radiative recombination continuum, i.e. (inf -> n=1) "Lyman limit"
#1260    H  1 3645.98A      radiative recombination continuum, i.e. (inf -> n=2) "Balmer limit"
#3552    H  1 1215.67A      H-like, 1 3,   1^2S -   2^2P, (n=2 to n=1) "Lyman-alpha" (first in Lyman-series)
#3557    H  1 1025.72A      H-like, 1 5,   1^2S -   3^2P, (n=3 to n=1) "Lyman-beta"
#3562    H  1 972.537A      H-like, 1 8,   1^2S -   4^2P, (n=4 to n=1) "Lyman-gamma"
#3672    H  1 6562.81A      H-like, 2 5,   2^2S -   3^2P, (n=3 to n=2) "H-alpha" / "Balmer-alpha"
#3677    H  1 4861.33A      H-like, 2 8,   2^2S -   4^2P, (n=4 to n=2) "H-beta" / "Balmer-beta"
#3682    H  1 4340.46A      H-like, 2 12,   2^2S -   5^2P, (n=5 to n=2) "H-gamma" / "Balmer-gamma"
#3687    H  1 4101.73A      H-like, 2 17,   2^2S -   6^2P, (n=6 to n=2) "H-delta" / "Balmer-delta"
#7487    C  6 33.7372A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (highest energy CVI line photon)
#7795    N  7 24.7807A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (")
#8103    O  8 18.9709A      H-like, 1 3,   1^2S -   2^2P, OVIII (n=2 to n=1) in Bertone+ 2010
#8108    O  8 16.0067A      H-like, 1 5,   1^2S -   3^2P, OVIII (n=3 to n=1)
#8113    O  8 15.1767A      H-like, 1 8,   1^2S -   4^2P, OVIII (n=4 to n=1)
#8148    O  8 102.443A      H-like, 2 5,   2^2S -   3^2P, OVIII (n=3 to n=2)
#8153    O  8 75.8835A      H-like, 2 8,   2^2S -   4^2P, OVIII (n=4 to n=2)
#8437    Ne10 12.1375A      H-like, 1 3,   1^2S -   2^2P, in vdV+ 2013
#8664    Na11 10.0250A      H-like, 1 3,   1^2S -   2^2P
#8771    Mg12 8.42141A      H-like, 1 3,   1^2S -   2^2P
#9105    Si14 6.18452A      H-like, 1 3,   1^2S -   2^2P
#9894    S 16 4.73132A      H-like, 1 3,   1^2S -   2^2P
#12819   Fe26 1.78177A      H-like, 1 3,   1^2S -   2^2P
#21954   C  5 40.2678A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#21989   C  5 41.4721A      He-like, 1 2,   1^1S -   2^3S
#23516   N  6 29.5343A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "resonance"
#24998   O  7 21.8070A      He-like, 1 5,   1^1S -   2^3P_1, in Bertone+ (2010) "intercombination"
#25003   O  7 21.8044A      He-like, 1 6,   1^1S -   2^3P_2, doublet? or effectively would be blend
#25008   O  7 21.6020A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#25043   O  7 22.1012A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "forbidden"
#26912   Ne 9 13.6987A      He-like, 1 2,   1^1S -   2^3S
#26867   Ne 9 13.5529A      He-like, 1 5,   1^1S -   2^3P_1
#26872   Ne 9 13.5500A      He-like, 1 6,   1^1S -   2^3P_2
#26877   Ne 9 13.4471A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#28781   Mg11 9.31434A      He-like, 1 2,   1^1S -   2^3S
#28736   Mg11 9.23121A      He-like, 1 5,   1^1S -   2^3P_1
#28741   Mg11 9.22816A      He-like, 1 6,   1^1S -   2^3P_2
#28746   Mg11 9.16875A      He-like, 1 7,   1^1S -   2^1P_1
#30650   Si13 6.74039A      He-like, 1 2,   1^1S -   2^3S
#30605   Si13 6.68828A      He-like, 1 5,   1^1S -   2^3P_1
#30610   Si13 6.68508A      He-like, 1 6,   1^1S -   2^3P_2
#30615   Si13 6.64803A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#32519   S 15 5.10150A      He-like, 1 2,   1^1S -   2^3S
#32474   S 15 5.06649A      He-like, 1 5,   1^1S -   2^3P_1
#32479   S 15 5.06314A      He-like, 1 6,   1^1S -   2^3P_2
#32484   S 15 5.03873A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#37124   Fe25 1.86819A      He-like, 1 2,   1^1S -   2^3S
#37079   Fe25 1.85951A      He-like, 1 5,   1^1S -   2^3P_1
#37084   Fe25 1.85541A      He-like, 1 6,   1^1S -   2^3P_2
#37089   Fe25 1.85040A      He-like, 1 7,   1^1S -   2^1P_1
#85082   C  3 1908.73A      Stout, 1 3
#85087   C  3 1906.68A      Stout, 1 4
#85092   C  3 977.020A      Stout, 1 5, in vdV+ 2013, in Bertone+ (2010b)
#123142  C  4 1550.78A      Chianti, 1 2, doublet in Bertone+ (2010b)
#123147  C  4 1548.19A      Chianti, 1 3, doublet in Bertone+ (2010b), in vdV+ 2013
#158187  O  6 1037.62A      Chianti, 1 2, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158192  O  6 1031.91A      Chianti, 1 3, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158197  O  6 183.937A      Chianti, 2 4
#158202  O  6 184.117A      Chianti, 3 4
#161442  S  4 1404.81A      Chianti, 1 3
#161447  S  4 1423.84A      Chianti, 2 3
#161452  S  4 1398.04A      Chianti, 1 4, in vdV+ 2013
#108822  O  2 3728.81A      Stout, 1 2, i.e. JWST/high-z emission line
#108827  O  2 3726.03A      Stout, 1 3, i.e. JWST/high-z emission line
#108847  O  3 4931.23A      Stout, 1 4, i.e. JWST/high-z emission line
#108852  O  3 4958.91A      Stout, 2 4, i.e. JWST/high-z emission line
#108857  O  3 5006.84A      Stout, 3 4, i.e. JWST/high-z emission line
#151382  N  2 6527.23A      Chianti, 1 4, i.e. JWST/high-z emission line
#151387  N  2 6548.05A      Chianti, 2 4, i.e. JWST/high-z emission line
#151392  N  2 6583.45A      Chianti, 3 4, i.e. JWST/high-z emission line
#110052  S  2 6730.82A      Stout, 1 2, i.e. JWST/high-z emission line
#110057  S  2 6716.44A      Stout, 1 3, i.e. JWST/high-z emission line
#110062  S  2 4076.35A      Stout, 1 4, i.e. JWST/high-z emission line
#110067  S  2 4068.60A      Stout, 1 5, i.e. JWST/high-z emission line
#167489  O  6 5291.00A      recombination line, i.e. inf -> n=
#167490  O  6 2082.00A      recombination line
#167491  O  6 3434.00A      recombination line
#167492  O  6 2070.00A      recombination line
#167493  O  6 1125.00A      recombination line
#229439  Blnd 2798.00A      Blend: "Mg 2      2795.53A"+"Mg 2      2802.71A"
#229562  Blnd 1035.00A      Blend: "O  6      1031.91A"+"O  6      1037.62A"
"""
# missing (for the future):
#Si  4 1393.755A, doublet in Bertone+ (2010b)
#Si  4 1402.770A, doublet in Bertone+ (2010b)
#N   5 1238.821A, doublet in Bertone+ (2010b)
#N   5 1242.804A, doublet in Bertone+ (2010b)
#Ne  8 770.409A, doublet in Bertone+ (2010b)
#Ne  8 780.324, doublet in Bertone+ (2010b)

def closest(array, value):
    """ Return closest element of array to input value. """
    ind = np.nanargmin( np.abs(array-value) )
    ind_nd = np.unravel_index( ind, array.shape )
    return array[ind_nd], ind

def iterable(x):
    """ Protect against non-list/non-tuple (e.g. scalar or single string) value of x, to guarantee that 
        a for loop can iterate over this object correctly. """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return np.reshape(x, 1) # scalar to 1d array of 1 element
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x        
    else:
        return [x]

def getEmissionLines():
    """ Return the list of emission lines (``lineList`` above) that we save from CLOUDY runs. """
    lines = lineList.split('\n')[1:-1] # first and last lines are blank in above string
    emLines = [line[9:22] for line in lines]
    wavelengths = [float(line[14:21]) for line in lines]

    return emLines, wavelengths

def resolveLineNames(lines, single=False):
    """ Map line abbreviations to unambiguous (species,ion,wavelength) triplets, leave inputs
    which are already full and valid unchanged. """
    emLines, _ = getEmissionLines()

    validLines = []
    for line in iterable(lines):
        if line in emLines:
            validLines.append(line)
            continue
        if line in lineAbbreviations:
            validLines.append(lineAbbreviations[line])
            continue
        if line.replace(" ","") in lineAbbreviations:
            validLines.append(lineAbbreviations[line.replace(" ","")])
            continue
        if line.replace(' ','-') in lineAbbreviations:
            validLines.append(lineAbbreviations[line.replace(' ','-')])
            continue
        raise Exception("Failed to recognize line [%s]!" % line)

    if single:
        # verify only a single line was input, and return a string not a list
        assert len(validLines) == 1
        return validLines[0]

    return validLines

def loadData(redshift, order, res, line=None):
    
    data    = {}
    grid    = {}
    Range   = {}

    with h5py.File(basePath + 'grid_emissivities_' + res + '.hdf5','r') as f:
        # load 4D line emissivity tables
        if line is None:
            lines = f.keys()
        else:
            lines = resolveLineNames(line)
            lines = [l.replace(" ","_") for l in iterable(lines)]

        for line in iterable(lines):
            data[line.replace("_"," ")] = f[line.replace("_"," ")][()]

        # load metadata/grid coordinates
        for attr in dict(f.attrs).keys():
            grid[attr] = f.attrs[attr]
    
    for field in grid.keys():
        Range[field] = [ grid[field].min(), grid[field].max() ]
        
    return(data, grid, Range)    

def emissivity(line, dens, metal, temp, redshift=0.0, order=3, res='lg'):
    """ Interpolate the line emissivity table for gas cell(s) with the given properties.
    Input gas properties can be scalar or np.array(), in which case they must have the same size.

    Args:
      line (str): name of line (species, ion number, and wavelength triplet) (or abbreviation).
      dens (float or None): hydrogen number density [cm^-3].
      temp (float or None): temperature [K].
      metal (float or None): metallicity [dimensionless; = M_z/ M_tot; the quantity saved in snapshots].
      redshift (float): redshift of the snapshot from which the above three files are derived.

    Return:
      ndarray: 1d array of volume emissivity, per cell [log erg/cm^3/s].
    """
    
    data, grid, Range = loadData(redshift, order, res, line)
    line = resolveLineNames(line, single=True)

    if line not in data:
        raise Exception('Requested line [' + line + '] not in grid.')
        
    # convert input interpolant point into fractional 3D/4D array indices
    # Note: we are clamping here at [0,size-1], which means that although we never 
    # extrapolate below (nearest grid edge value is returned), there is no warning given
    i1 = np.interp( np.log10(dens),  grid['dens'],  np.arange(grid['dens'].size) )
    i2 = np.interp( np.log10(metal/0.0127), grid['metal'], np.arange(grid['metal'].size) )
    i3 = np.interp( np.log10(temp),  grid['temp'],  np.arange(grid['temp'].size) )

    i0 = np.interp( redshift, grid['redshift'], np.arange(grid['redshift'].size) )
    if isinstance(i0,float):
        i0 = np.zeros( i1.shape, dtype='float32' ) + i0 # expand scalar into 1D array

    iND = np.vstack( (i0,i1,i2,i3) )

    # do 3D or 4D interpolation on this ion sub-table at the requested order
    locData = data[line]

    emis = map_coordinates( locData, iND, order=order, mode='nearest')

    return emis
