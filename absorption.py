"""
Class to compute optical depth for the broad line region
from Finke (2016)
"""

# --- Imports -------------------- #
import numpy as np
import fsrq
from scipy.integrate import simps
from astropy import units as u
from astropy import constants as c
from astropy.table import Table
from os import path
# -------------------------------- #
def floatList2ndarray(x):
    """Convert list or scalar to numpy array"""
    if np.isscalar(x):
	return np.array([x])
    elif type(x) == list:
	return np.array(x)
    elif type(x) == np.ndarray:
	return x
    raise ValueError("Type {0} not supported".format(type(x)))

def L5100(LHbeta, norm = 1.425e42, index = 0.8826):
    """
    Compute the luminosity at 5100 Angstrom from the luminosity of the 
    H beta line using the relation from Greene & Ho (2005)

    Parameters
    ----------
    LHbeta: float or `~numpy.ndarray`
	Luminosity of H beta line in erg / s

    Returns
    -------
    float float or `~numpy.ndarray`, luminosity at 5100 Angstrom in ergs / s.

    Note
    ----
    uncertainty of norm is 0.007e42 and on the index it is 0.0039
    """
    return np.power(LHbeta / norm, index) * 1e44

def RHbeta(L5100, norm = 16.94, index = 0.533):
    """
    Compute the radius where H beta is primarlity emitted from 
    luminosity at 5100 Angstrom 
    using the relation from Bentz et al. (2013)

    Parameters
    ----------
    LHbeta: float or `~numpy.ndarray`
	Luminosity at 5100 Angstrom in erg / s

    Returns
    -------
    float float or `~numpy.ndarray`, radius in cm

    Note
    ----
    uncertainty of norm is 0.03 and on the index it is 0.035
    """
    return np.power(10.,norm) * np.power(L5100 / 1e44, index)

def sgg(s):
    """
    Gamma-gamma cross section 

    Parameters
    ----------
    s: float or `~numpy.ndarray`
	center of mass energy

    Returns
    -------
    float or `~numpy.ndarray` with cross section in units of Thomson cross section, 
    i.e. sigma_gammagamma / sigma_T

    Notes
    -----
    see e.g. Finke 2016, Eq. 119
    """
    beta = np.zeros(s.shape)
    m = s >= 1.
    beta[m] = np.sqrt(1. - 1./s[m])


    result = np.zeros(beta.shape)
    result[m] = (1. - beta[m] * beta[m]) \
    		* ( (3. - beta[m] * beta[m] * beta[m] * beta[m]) \
		* (np.log(1. + beta[m]) - np.log(1. - beta[m])) \
		- 2. * beta[m] * (2. - beta[m] * beta[m]))
    #return 3. / 8. * result
    return 3. / 16. * result

class BLR(object):
    def __init__(self, Ldisk, z, lumiHbeta = 4.2e43, radiusHbeta = 0., lumi5100 = 0., M_BH = 1e9):
	"""
	Initiate the class

	Parameters
	----------
	Ldisk: float
	    disk luminosity in erg/s

	lumiHbeta: float
	    Luminosity of H beta line in erg/s  (default : 0.)

	z: float
	    redshift of the source

	{options}

	M_BH: float
	    mass of central supermassive black hole in solar masses (default : 1e9)

	lumi5100: float or None
	    Luminosity at 5100 Angstrom (default : 0.). If 0., will be calculated 
	    from LumiHbeta.

	radiusHbeta: float
	    radius from where majority of HBeta photons are emitted (default 0.).
	    If 0., will be calculated from L5100 luminosity
	"""
	self._Ldisk = Ldisk
	self._M_BH = M_BH
	self._blr_lines = Table.read(path.join(path.dirname(fsrq.__file__), 'blr_line_pars.fits'))
	self._z = z

	if not lumi5100: 
	    if not lumiHbeta:
		raise ValueError("If L5100 not given, LHbeta must be > 0!")
	    self._lumi5100 = L5100(lumiHbeta)
	else:
	    self._lumi5100 = lumi5100

	if not radiusHbeta:
	    self._radiusHbeta = RHbeta(self._lumi5100)
	else:
	    self._radiusHbeta = radiusHbeta

	# Eddington luminosity in erg/s
	self._Ledd = (4. * np.pi * c.G * M_BH * c.M_sun * \
		    c.m_p * c.c / c.sigma_T).to('erg / s').value
	# gravitational radius in cm
	self._Rg = (c.G * M_BH * c.M_sun / c.c / c.c).to('cm').value 

	# Energy of emission lines
	self._Eli = (c.h * c.c / self._blr_lines['Wavelength'].to('m')).to('eV').value

	# Radii of emission lines
	self._Rli = self._radiusHbeta * self._blr_lines['Radius']

	# Energy of lines in units of electron mass
	self._me = (c.m_e * c.c * c.c).to('eV').value
	self._epsli = self._Eli / self._me

	# line luminosities 
	self._Lli = lumiHbeta * self._blr_lines['Luminosity']
	self._xili = self._Lli / self._Ldisk
	return

    @property
    def Ldisk(self):
	return self._Ldisk

    @property
    def z(self):
	return self._z

    @property
    def blr_lines(self):
	return self._blr_lines

    @blr_lines.setter
    def blr_lines(self, fits_file):
    	self._blr_lines = Table.read(fits_file)
	# Energy of emission lines
	self._Eli = (c.h * c.c / self._blr_lines['Wavelength'].to('m')).to('eV').value
	return self._blr_lines

    @property
    def M_BH(self, M_BH):
	return self._M_BH

    @M_BH.setter
    def M_BH(self):
    	self._M_BH = M_BH
	# Eddington luminosity in erg/s
	self._Ledd = (4. * np.pi * c.G * M_BH * c.M_sun * \
		    c.m_p * c.c / c.sigma_T).to('erg / s').value
	# gravitational radius in cm
	self._Rg = (c.G * M_BH * c.M_sun / c.c / c.c).to('cm').value 
	return 

    @property
    def Rli(self):
	return self._Rli

    @Rli.setter
    def Rli(self, Rli):
    	self._Rli = Rli
	return 

    @property
    def Lli(self):
	return self._Lli

    @property
    def xili(self):
	return self._xili

    @xili.setter
    def xili(self, xili):
	self._xili = xili
	return 

    @property
    def Eli(self):
	return self._Eli

    @Eli.setter
    def Eli(self, Eli):
	self._Eli = Eli
	self._epsli = Eli / self._me
	return 

    def __xtilde_sq_shell(self, Rli, l, mu_re):
	res = (Rli * Rli + l*l - 2. * l * Rli * mu_re )/ self._Rg / self._Rg
	if not np.isscalar(res):
	    res[res == 0.] = np.ones(np.sum(res == 0.)) * 1e-10
	else: 
	    if res == 0.: return 1e-10
	return res

    def __mu_star_sq_shell(self, Rli, mu_re, l):
    	res = 1. - Rli * Rli / self._Rg / self._Rg / self.__xtilde_sq_shell(Rli, l, mu_re) \
		* (1. - mu_re * mu_re)

	return res

    def __stilde_shell(self, epsli, eps, Rli, mu_re, l):
	mustar_sq = self.__mu_star_sq_shell(Rli, mu_re, l)
	return 0.5 * epsli * eps * (1. + self.z) \
		    * (1. - np.sqrt(mustar_sq))
    def stilde_shell(self, epsli, eps, Rli, mu_re, l):
	return self.__stilde_shell(epsli, eps, Rli, mu_re, l)

    def __xtilde_sq_ring(self, Rli, ltilde):
	return Rli * Rli / self._Rg / self._Rg + ltilde * ltilde

    def __stilde_ring(self, epsli, eps, ltilde, Rli):
	return 0.5 * epsli * eps * (1. + self.z) * \
		    (1. - ltilde / np.sqrt(self.__xtilde_sq_ring(Rli, ltilde)))
	


    def gamma_abs(self, epsli, eps, Rli, l, mu = None, geometry = 'shell', 
	value = True, xili = None):
	"""
	Absorption length
	for gamma-gamma attenuation

	Parameters
	----------
	epsli: `~numpy.ndarray`
	    energy of target photon field in units of electron mass

	eps: `~numpy.ndarray`
	    gamma-ray energy of in units of electron mass

	Rli:  `~numpy.ndarray` 
	    distance of target photon field to central BH
	
	l:  `~numpy.ndarray`
	    distance of interaction point to central BH


	{options}

	geometry: str
	    geometry of target photon field, either 'shell' or 'ring'
	
	mu:  `~numpy.ndarray` or None
	    if geometry == 'shell', mu gives angle between photons
	
	value: bool
	    if True, return interaction length in units of cm. 
	    If False, return 1 / interaction legth w/o units (used for integration)
		
	"""
	if geometry == 'shell':
	    stilde = self.__stilde_shell(epsli, eps, Rli, mu, l)
	    xtilde_sq = self.__xtilde_sq_shell(Rli, l, mu)
	    mustar_sq = self.__mu_star_sq_shell(Rli, mu, l)

	    kernel = sgg(stilde) * ( 1. - np.sqrt(mustar_sq) ) / xtilde_sq
	    result = simps(kernel, mu, axis = -1)

	elif geometry == 'ring':
	    stilde = self.__stilde_ring(epsli, eps, l, Rli)
	    xtilde_sq = self.__xtilde_sq_ring(Rli, l)

	    result = sgg(stilde) / xtilde_sq * (1. - l / np.sqrt(xtilde_sq))

	elif geometry == 'extended':
	    # mu is now the integration over the extended dust torus R coordinate
	    xtilde_sq = self.__xtilde_sq_ring(mu, l)
	    stilde = self.__stilde_ring(epsli, eps, l, mu)
	    kernel = sgg(stilde) / xtilde_sq * (1. - l / np.sqrt(xtilde_sq))
	    result = simps(kernel * mu, np.log(mu), axis = -1)

	if value:
	    gamma = np.zeros(result.shape)
	    gamma[result > 0.] = self._Rg / (900. * xili * self._Ldisk / self._Ledd \
			/ epsli * result[result > 0.])
	    return gamma
	else: 
	    return result

    def tau(self, EGeV, Rcm, lsteps = 100, mu_steps = 101, rtilde_max = 1e6, 
    		geometry = 'shell'):
	"""
	Calculate the absorption on a BLR with shell geometry on all lines
	included in blr_lines table

	Parameters:
	----------
	EGeV: float or `~numpy.ndarray`
	    Gamma-ray energy in GeV, n-dimensional

	Rcm: float or `~numpy.ndarray`
	    radius of gamma-ray emitting region in cm, m-dimensional

	Returns
	-------
	n x m dimensional `~numpy.ndarray` with gamma-ray optical depth
	"""
	# convert to numpy arrays
	EGeV = floatList2ndarray(EGeV)
	Rcm = floatList2ndarray(Rcm)

	eps = ((EGeV * u.GeV).to('eV') / self._me / u.eV).value

	if geometry == 'shell':
	    mu_re = np.linspace(-1.,1.,mu_steps)

	    ee, rr, ll, mm = np.meshgrid(eps, Rcm / self._Rg, np.arange(lsteps, dtype = np.float), 
					mu_re,  indexing = 'ij')

	    for i in range(eps.shape[0]): 
		for j,r in enumerate(Rcm): 
		    for k in range(mu_re.shape[0]): 
			ll[i,j,:,k] = np.logspace(np.log10(r /  self._Rg), np.log10(rtilde_max) , lsteps)

	elif geometry == 'ring': 
	    ee, rr, ll = np.meshgrid(eps, Rcm / self._Rg, np.arange(lsteps, dtype = np.float), 
					indexing = 'ij')
	    for i in range(eps.shape[0]): 
		for j,r in enumerate(Rcm): 
			ll[i,j,:] = np.exp(np.linspace(np.log(r /  self._Rg), np.log(rtilde_max) , lsteps))

	elif geometry == 'extended': 
	    Rdt = np.logspace(np.log10(self._Rdt_in), np.log10(self._Rdt_out), mu_steps)

	    ee, rr, ll, mm = np.meshgrid(eps, Rcm / self._Rg, np.arange(lsteps, dtype = np.float), 
					Rdt,  indexing = 'ij')

	    for i in range(eps.shape[0]): 
		for j,r in enumerate(Rcm): 
		    for k in range(Rdt.shape[0]): 
			ll[i,j,:,k] = np.logspace(np.log10(r /  self._Rg), np.log10(rtilde_max) , lsteps)

	else:
	    raise ValueError("Unknown geometry chosen: {0}".format(geometry))

	# loop through all lines
	tau = np.zeros(eps.shape + Rcm.shape + self._epsli.shape)
	for i, epsli in enumerate(self._epsli):

	    kernel = self.gamma_abs(epsli, ee, self._Rli[i], 
			    ll * self._Rg if geometry == 'shell' else ll,
			    mu = None if geometry == 'ring' else mm,  
			    geometry = geometry, value = False)

	    if geometry == 'shell' or  geometry == 'extended':
		result = simps(kernel * ll[...,-1], np.log(ll[...,-1]), axis = -1)

	    elif geometry == 'ring':
		result = simps(kernel * ll, np.log(ll), axis = -1)

	    tau[...,i] = 900. * self.xili[i] * self._Ldisk / self._Ledd / epsli * result 

	    if geometry == 'extended':
		tau[...,i] /= (self._Rdt_eff / self._Rg)

	return np.squeeze(tau)


class DustTorus(BLR):
    def __init__(self, Ldisk, z, 
	xidt = 0.1, Tdust = 1e3, 
	Rdt_in = 3.5e18, Rdt_out = 3.5e19, 
	zeta = 1.,
	M_BH = 1.2e9 ):
	"""
	Initiate class for absorption on dust torus radiation field

	Parameters
	----------
	Ldisk: float
	    disk luminosity in erg/s

	z: float
	    redshift of the source

	{options}

	M_BH: float
	    mass of central supermassive black hole in solar masses (default : 1e9)

	Tdust: float
	    Dust black body temperature in K

	Rdt_in: float
	    Inner dust radius in cm

	Rdt_out: float
	    Outer dust radius in cm

	xidt: float
	    Dust torus scattering fraction

	zeta: float
	    Dust torus scattering fraction
	"""
	self._Tdust = Tdust
	self._Rdt_in = Rdt_in
	self._Rdt_out = Rdt_out
	self._zeta = zeta 

	super(DustTorus, self).__init__( Ldisk, z, M_BH = M_BH, radiusHbeta = Rdt_in)

	# setting Eli also sets epsli
	self.Eli = (2.7 * np.array([Tdust]) * u.K * c.k_B).to('eV').value
	self.xili = np.array([xidt])
	#self.Rli = np.array([Rdt_in])
	self.Rli = np.array([Rdt_out])
	if zeta == 1.:
	    self._Rdt_eff = Rdt_in * (np.log(Rdt_out) - np.log(Rdt_in))
	else:
	    self._Rdt_eff = (Rdt_in - Rdt_out * (Rdt_out / Rdt_in)**-zeta) \
			    / (zeta - 1.)

	return

class AccretionDisk(BLR):
    def __init__(self, Ldisk, z, 
	M_BH = 1.2e9, eta = 1. / 12.,  
	R_out = 200., 
	a = 0.
	):
	"""
	Initiate class for absorption on accretion disk photons

	Parameters
	----------
	Ldisk: float
	    disk luminosity in erg/s

	z: float
	    redshift of the source

	{options}

	M_BH: float
	    mass of central supermassive black hole in solar masses (default : 1e9)

	a: float,
	    BH spin

	eta: float
	    Accretion efficiency

	R_in: float
	    Inner disk radius in gravitational radii

	R_out: float
	    Outer disk radius in gravitational radii
	"""
	self._a = a
	self._eta = eta
	self.R_out = R_out

	super(AccretionDisk, self).__init__( Ldisk, z, M_BH = M_BH)
	return 

    def __A1(self):
	return 1. + (1. - self._a * self._a) ** (1. / 3.) \
		    * ((1. + self._a) ** (1. / 3.) + (1. - self._a)**(1. / 3.))
    def __A2(self):
	return np.sqrt(3. * self._a * self._a + self.__A1() * self.__A1())

    def RinTilde(self): 
	"""Innermost radius in units of gravitational radius"""
	if self._a == 0.:
	    return 6.
	else:
	    return 3. + self.__A2() - \
		np.abs(self._a) / self._a * ((3. - self.__A1()) * (3. + self.__A1() + 2. * self.__A2()))

    def eps0(self, Rtilde): 
	return 2.7e-4 * (self._Ldisk / self._Ledd / self._M_BH / self._eta * 1e8) ** 0.25 \
	    * Rtilde ** -0.75

    def __Phi(self, Rtilde):
	return np.sqrt(1. - self.RinTilde() / Rtilde)

    def __mu(self, Rtilde, ltilde):
	return 1. / np.sqrt(1. + Rtilde / ltilde)


    def __stilde_disk(self, eps, Rtilde, ltilde):
	return 0.5 * self.eps0(Rtilde) * eps * (1. - self._z) * (1. - self.__mu(Rtilde, ltilde))

    def gamma_abs_disk(self, eps, Rtilde, ltilde, value = True): 
	"""
	Absorption length
	for gamma-gamma attenuation

	Parameters
	----------
	eps: `~numpy.ndarray`
	    gamma-ray energy of in units of electron mass

	Rtilde:  `~numpy.ndarray` 
	    radius within accretion disk in units of gravitational radius
	
	ltilde:  `~numpy.ndarray`
	    distance of interaction point to central BH in 
	    units of gravitatinal radius


	{options}

	value: bool
	    if True, return interaction length in units of cm. 
	    If False, return 1 / interaction legth w/o units (used for integration)
		
	"""
	stilde = self.__stilde_disk(eps, Rtilde, ltilde)
	mu = self.__mu(Rtilde, ltilde)
	phi = self.__Phi(Rtilde)


	kernel = sgg(stilde) * phi / Rtilde ** 1.25 \
		/ (1. + (Rtilde / ltilde) * (Rtilde / ltilde)) ** 1.5 \
		* ( 1. - mu ) 

	result = simps(kernel * Rtilde, np.log(Rtilde), axis = -1)

	if value:
	    return result * self._Rg / (1e7 * (self._Ldisk / self._Ledd / self._eta ) ** 0.75 \
			* (self._M_BH / 1e8) ** 0.25 )
	else: 
	    return result

    def tau_disk(self, EGeV, Rcm, lsteps = 100, rsteps = 101, ltilde_max = 1e6):
	"""
	Calculate the absorption on accretion disk photons

	Parameters:
	----------
	EGeV: float or `~numpy.ndarray`
	    Gamma-ray energy in GeV, n-dimensional

	Rcm: float or `~numpy.ndarray`
	    radius of gamma-ray emitting region in cm, m-dimensional

	Returns
	-------
	n x m dimensional `~numpy.ndarray` with gamma-ray optical depth
	"""
	EGeV = floatList2ndarray(EGeV)
	Rcm = floatList2ndarray(Rcm)

	eps = ((EGeV * u.GeV).to('eV') / self._me / u.eV).value

	Rtilde = np.logspace(np.log10(self.RinTilde()),np.log10(self.R_out),rsteps)

	ee, rr, ll, RR = np.meshgrid(eps, Rcm / self._Rg, np.arange(lsteps, dtype = np.float), 
					Rtilde,  indexing = 'ij')

	for i in range(eps.shape[0]): 
	    for j,r in enumerate(Rcm): 
		for k in range(Rtilde.shape[0]): 
		    ll[i,j,:,k] = np.logspace(np.log10(r /  self._Rg), np.log10(ltilde_max) , lsteps)


	kernel = self.gamma_abs_disk(ee, RR, ll, value = False) / ll[...,-1]
	result = simps(kernel, np.log(ll[...,-1]), axis = -1)

	tau = 1e7 * (self._Ldisk / self._Ledd / self._eta ) ** 0.75 \
			* (self._M_BH / 1e8) ** 0.25 * result 

	return np.squeeze(tau)
