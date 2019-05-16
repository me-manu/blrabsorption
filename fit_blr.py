# --- imports ------ #
import numpy as np
import iminuit as minuit 
import time
import functools
import logging
from scipy.stats import norm,lognorm
from scipy.special import ndtri,erf
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.interpolate import UnivariateSpline as USpline
from scipy import optimize as op
from collections import OrderedDict
from copy import deepcopy
from astropy.table import Table
# ------------------ #

# --- Helper functions ------------------------------------- #
erfnorm = lambda z1,z0,x0,sigma: 0.5 * \
                    (erf((z1 - x0)/np.sqrt(2. * sigma**2.)) - \
                     erf((z0 - x0)/np.sqrt(2. * sigma**2.)))

def setDefault(func = None, passed_kwargs = {}):
    """
    Read in default keywords of the simulation and pass to function
    """
    if func is None:
	return functools.partial(setDefault, passed_kwargs = passed_kwargs)
    @functools.wraps(func)
    def init(*args, **kwargs):
	for k in passed_kwargs.keys():
	    kwargs.setdefault(k,passed_kwargs[k])
	return func(*args, **kwargs)
    return init
# ---------------------------------------------------------- #

# --- miniuit defaults ------------------------------------- #
#The default tolerance is 0.1. 
#Minimization will stop when the estimated vertical distance to the minimum (EDM) 
#is less than 0.001*tolerance*UP (see SET ERR). 

minuit_def = {
	'verbosity': 0,	
	'int_steps': 1e-4,
	'strategy': 2,
	'tol': 1e-5,
	'up': 0.5,
	'max_tol_increase': 3000.,
	'tol_increase': 1000.,
	'ncall': 5000,
	'r_steps': 40,
	'scan_bound': (16.,18.),
	'pedantic': True,
	'precision': None,
	'scipy': True,
	'pinit': {'norm' : -10.,
		'index': -3.,
		'alphac': 1.,
		'r': 17.},
	'fix': {'norm' : False,
		'index': False,
		'alphac': False,
		'r': False },
	'limits': {'norm' : [-20,-5],
		'index': [-5,5],
		'alphac': [0.1,10.],
		'r': [16.,18.]},
	'islog': {'norm' : True,
		'index': False,
		'alphac': False,
		'r': True},
	}
# --- miniuit defaults ------------------------------------- #

class FitBLR(object):
    def __init__(self, llh, taublr, EGeVllh, EGeVtau, fluxllh, rtau,
    		kx = 2, ky = 2):
        """
        Initialize the class

        Parameters
        ----------
        llh: `~numpy.ndarray`
            n x m dimensional cube, dimension n: energy bins, dimension m: flux bins, 
            each entry gives the log likelihood value for that energy bin and flux

        taublr: `~numpy.ndarray`
            i x k dimensional cube with optical depths for i energies and k distances

        EGeVllh:`~numpy.ndarray`
            n dimensional array with central bin energies for llh cube

        EGeVtau:`~numpy.ndarray`
            i dimensional array with central bin energies for tau cube

        flux_llh:`~numpy.ndarray`
            m dimensional array with central flux bin values for llh cube
            or n x m dimensional array with central flux bin values for llh cube
            for each energy bin
            
        rtau :`~numpy.ndarray`
            k dimensional array with central distance bin values for tau cube
        """
        self._taublr = taublr
        self._EGeVllh = EGeVllh
        self._fluxllh = fluxllh
        self._EGeVtau = EGeVtau
        self._rtau = rtau
        self._llh = llh
        self._llh_intp = []

# interpolate taublr
        self.__tauSpline = RBSpline(np.log(EGeVtau),np.log(rtau),self._taublr,kx=kx,ky=ky)

# piece-wise interpolation of llh 
        for i,l in enumerate(self._llh):
            if len(self._fluxllh.shape) == 1:
                self._fluxllh[i][self._fluxllh == 0.] = \
                    1e-40 * np.ones(np.sum(self._fluxllh == 0.))
                self._llh_intp.append(USpline(np.log(self._fluxllh),l,
                    k = 2, s = 0, ext = 'extrapolate'))
            elif len(self._fluxllh.shape) == 2:
                self._fluxllh[i][self._fluxllh[i] == 0.] = \
                    1e-40 * np.ones(np.sum(self._fluxllh[i] == 0.))
                self._llh_intp.append(USpline(np.log(self._fluxllh[i]),l,
                    k = 2, s = 0, ext = 'extrapolate'))
        return 

    @staticmethod
    def readfermised(sedfile, taublr, EGeVtau, rtau, hdu = 1, tsmin = None, emin = None, emax = None, fit_qual = False):
        """
        Initialize the class by building a data likelihood cube 
        from a fermipy SED fits file

        Parameters
        ----------
        sedfile: string
            full path to fermi sed fits file

        taublr: `~numpy.ndarray`
            i x k dimensional cube with optical depths for i energies and k distances

        EGeVtau:`~numpy.ndarray`
            i dimensional array with central bin energies for tau cube

        rtau :`~numpy.ndarray`
            k dimensional array with central distance bin values for tau cube

        tsmin: float
            minimum ts value for an energy bin to be considered in the fit
            default: None

        emin: float
            minimum energy in MeV for an energy bin to be considered in the fit
            default: None

        emax: float
            maximum energy in MeV for an energy bin to be considered in the fit
            default: None

        fit_qual: bool
            if True, use only energy bins that have fit quality == 3. and status == 0.
            (only works if sed provided in npy file)
            (only works if sed provided in npy file)
        """
        if 'fits' in sedfile:
            t = Table.read(sedfile, hdu = hdu)
        elif 'npy' in sedfile:
            t = np.load(sedfile).item()
        else:
            raise IOError("Input file type not understood")

        #llh = t['dloglike_scan'].data
        if type(tsmin) == float:
            m = t['ts'] > tsmin
        else:
            m = np.ones_like(t['ts'], dtype = np.bool)
        if 'npy' in sedfile and fit_qual:
            m = m & (t['fit_quality'] == 3.) & (t['fit_status'] == 0.)

        if type(emin) == float:
            m &= t['e_min'] >= emin
        if type(emax) == float:
            m &= t['e_max'] <= emax

        if 'fits' in sedfile:
            llh = (t['dloglike_scan'].data[m].T + t['loglike'][m]).T # ebins x flux dimensions
            fluxcube = (t['ref_dnde'].data[m] * t['norm_scan'].data[m].T).T # ebins x flux dimensions
            EGeVdata = t['e_ref'].data[m] * 1e-3
        else:
            llh = (t['dloglike_scan'][m].T + t['loglike'][m]).T # ebins x flux dimensions
            fluxcube = (t['ref_dnde'][m] * t['norm_scan'][m].T).T # ebins x flux dimensions
            EGeVdata = t['e_ref'][m] * 1e-3

        return FitBLR(llh, taublr, EGeVdata, EGeVtau, fluxcube, rtau)

    @staticmethod
    def readxydata(x,y,dy, mUL, taublr, EGeVtau, rtau, scale = 5., ysteps = 400, ymin = 1e-15, 
    			ULconf = 0.95, distr = 'norm'):
        """
        Initialize the class by building a data likelihood cube 
        from x (EGeV) and y (dN/dE) data with statistical error dy

        Parameters
        ----------
        x: `~numpy.ndarray`
            n-dim array with energy values in GeV

        y: `~numpy.ndarray`
            n-dim array with flux values 

        dy: `~numpy npdarray`
            n-dim array with statistical uncertainties on flux values

        mUL: `~numpy npdarray`
            n-dim array with bools that are true if y in the bin is upper limit

        taublr: `~numpy.ndarray`
            i x k dimensional cube with optical depths for i energies and k distances

        EGeVtau:`~numpy.ndarray`
            i dimensional array with central bin energies for tau cube

        rtau :`~numpy.ndarray`
            k dimensional array with central distance bin values for tau cube

        {options}

        scale: float
            numbers of sigmas for likelihood scan
        
        ysteps: int
            number of steps for likelihood scan

        ymin: float
            minimum flux assumed for likelihood scan
        
        ULconf: float
            confidence interval for upper limit
        """
        ymin = np.min((ymin, np.min(np.abs(y - scale * dy))))
        ymax = np.max(y + scale * dy)
        yrange = np.logspace(np.log10(ymin), np.log10(ymax), ysteps)

        llh = []
        for i,E in enumerate(x):
            if mUL[i]:
                llh.append(norm.pdf(yrange, loc = ymin, scale = dy[i] / ndtri(ULconf)))
                print "assuming {0:.3f} confidence level".format(
                    erfnorm(dy[i], 0.,0., dy[i] / ndtri(ULconf)) + 0.5
                    )
            else:
                if distr == 'norm':
                    llh.append(norm.pdf(yrange, loc = y[i], scale = dy[i]))
                elif distr == 'lognorm':
                    sigma = np.sqrt(np.log(1. + dy[i]**2. / y[i]**2.))
                    llh.append(lognorm.pdf(yrange / y[i], sigma))

            llh[-1][llh[-1] < 1e-200] = np.ones(np.sum(llh[-1] < 1e-200)) * 1e-200

        return FitBLR(np.log(llh), taublr, x, EGeVtau, yrange, rtau)

    def opt_depth(self,rcm,EGeV):
        """
        Returns optical depth for distance r in cm and Engergy (GeV) from BSpline Interpolation 

        Parameters
        ----------
        rcm: `~numpy.ndarray` or list
            distance of gamma-ray emitting region to BH in cm, m-dimensional

        EGeV: `~numpy.ndarray` or list
            Energies in GeV, n-dimensional

        Returns
        -------
        (N x M) `~numpy.ndarray` with corresponding optical depth values.
        If rcm or EGeV are scalars, the corresponding axis will be squezed.

        """
        if np.isscalar(EGeV):
            EGeV = np.array([EGeV])
        elif type(EGeV) == list:
            EGeV = np.array(EGeV)
        if np.isscalar(rcm):
            rcm = np.array([rcm])
        elif type(rcm) == list:
            rcm = np.array(rcm)

        result = np.zeros((rcm.shape[0],EGeV.shape[0]))
        tt = np.zeros((rcm.shape[0],EGeV.shape[0]))

        args_r = np.argsort(rcm)
        args_E = np.argsort(EGeV)

        # Spline interpolation requires sorted lists
        tt[args_r,:] = self.__tauSpline(np.log(np.sort(EGeV)),np.log(np.sort(rcm))).transpose()	
        result[:,args_E] = tt

        return np.squeeze(result)

    def calcLikelihood(self,*args):
    	return self.__calcLikelihood(*args)

    def __calcLikelihood(self,*args):
        """
        likelihood function passed to iMinuit
        """
        params = {}
        for i,p in enumerate(self.parnames):
            if self.par_islog[p]:
                params[p] = np.power(10.,args[i])
            else:
                params[p] = args[i]
        return self.returnLikelihood(params)

    def __wrapLikelihood(self,args):
        """
        likelihood function passed to scipy.optimize 
        """
        params = {}
        for i,p in enumerate(self.parnames):
            if not self.fitarg['fix_{0:s}'.format(p)]:
                if self.par_islog[p]:
                    params[p] = np.power(10.,args[i])
                else:
                    params[p] = args[i]
            else:
                if self.par_islog[p]:
                    params[p] = np.power(10.,self.fitarg[p])
                else:
                    params[p] = self.fitarg[p]
        return self.returnLikelihood(params)

    def returnLikelihood(self,params):
        """Calculate the log likelihood"""


        f = self._int_spec(self._EGeVllh, **params)
        tau = self.opt_depth(params['r'],self._EGeVllh)

        llh = 0.
        self.llh_pw = np.zeros(self._EGeVllh.shape[0])
        for i,x in enumerate(self._EGeVllh):
            llh += self._llh_intp[i](np.log(f[i]) - tau[i])
            self.llh_pw[i] = self._llh_intp[i](np.log(f[i]) - tau[i])

        return -1. * llh

    @setDefault(passed_kwargs = minuit_def)
    def fill_fitarg(self, **kwargs):
        """
        Helper function to fill the dictionary for minuit fitting
        """
        # set the fit arguments
        fitarg = {}
        fitarg.update(kwargs['pinit'])
        for k in kwargs['limits'].keys():
            fitarg['limit_{0:s}'.format(k)] = kwargs['limits'][k]
            fitarg['fix_{0:s}'.format(k)] = kwargs['fix'][k]
            fitarg['error_{0:s}'.format(k)] = kwargs['pinit'][k] * kwargs['int_steps']

        fitarg = OrderedDict(sorted(fitarg.items()))
        # get the names of the parameters
        self.parnames = kwargs['pinit'].keys()
        self.par_islog = kwargs['islog']
        return fitarg

    @setDefault(passed_kwargs = minuit_def)
    def run_migrad(self,fitarg,**kwargs):
        """
        Helper function to initialize migrad and run the fit.
        Initial parameters are estimated with scipy fit.
        """
        self.fitarg = fitarg

        values, bounds = [],[]
        for k in self.parnames:
            values.append(fitarg[k])
            bounds.append(fitarg['limit_{0:s}'.format(k)])

        logging.info(self.parnames)
        logging.info(values)

        logging.info(self.__wrapLikelihood(values))

        if kwargs['scipy']:
            self.res = op.minimize(self.__wrapLikelihood, 
                values, 
                bounds = bounds,
                method='TNC',
                #method='Powell',
                options={'maxiter': kwargs['ncall']} #'xtol': 1e-20, 'eps' : 1e-20, 'disp': True}
                #tol=None, callback=None, 
                #options={'disp': False, 'minfev': 0, 'scale': None, 
		    		#'rescale': -1, 'offset': None, 'gtol': -1, 
                    #'eps': 1e-08, 'eta': -1, 'maxiter': kwargs['ncall'], 
                    #'maxCGit': -1, 'mesg_num': None, 'ftol': -1, 'xtol': -1, 'stepmx': 0, 
                    #'accuracy': 0}
                )
            logging.info(self.res)
            for i,k in enumerate(self.parnames):
                fitarg[k] = self.res.x[i]

            logging.debug(fitarg)
        cmd_string = "lambda {0}: self.__calcLikelihood({0})".format(
					(", ".join(self.parnames), ", ".join(self.parnames)))

        string_args = ", ".join(self.parnames)
        global f # needs to be global for eval to find it
        f = lambda *args: self.__calcLikelihood(*args)
	
        cmd_string = "lambda %s: f(%s)" % (string_args, string_args)
        logging.debug(cmd_string)

        # work around so that the parameters get names for minuit
        self._minimize_f = eval(cmd_string, globals(), locals())

        self.m = minuit.Minuit(self._minimize_f, 
            print_level =kwargs['verbosity'],
            errordef = kwargs['up'], 
            pedantic = kwargs['pedantic'],
            **fitarg)

        self.m.tol = kwargs['tol']
        self.m.strategy = kwargs['strategy']

        logging.debug("tol {0:.2e}, strategy: {1:n}".format(
               self.m.tol,self.m.strategy))

        self.m.migrad(ncall = kwargs['ncall']) #, precision = kwargs['precision'])
        return 

    def __print_failed_fit(self):
        """print output if migrad failed"""
        if not self.m.migrad_ok():
            fmin = self.m.get_fmin()
            logging.warning(
            '*** migrad minimum not ok! Printing output of get_fmin'
            )
            logging.warning('{0:s}:\t{1}'.format('*** has_accurate_covar',
            fmin.has_accurate_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_covariance',
            fmin.has_covariance))
            logging.warning('{0:s}:\t{1}'.format('*** has_made_posdef_covar',
            fmin.has_made_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_posdef_covar',
            fmin.has_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_reached_call_limit',
            fmin.has_reached_call_limit))
            logging.warning('{0:s}:\t{1}'.format('*** has_valid_parameters',
            fmin.has_valid_parameters))
            logging.warning('{0:s}:\t{1}'.format('*** hesse_failed',
            fmin.hesse_failed))
            logging.warning('{0:s}:\t{1}'.format('*** is_above_max_edm',
            fmin.is_above_max_edm))
            logging.warning('{0:s}:\t{1}'.format('*** is_valid',
            fmin.is_valid))
        return


    def __repeat_migrad(self, **kwargs):
        """Repeat fit if fit was above edm"""
        fmin = self.m.get_fmin()
        if not self.m.migrad_ok() and fmin['is_above_max_edm']:
            logging.warning(
            'Migrad did not converge, is above max edm. Increasing tol.'
            )
            tol = self.m.tol
            self.m.tol *= self.m.edm /(self.m.tol * self.m.errordef ) * kwargs['tol_increase']

            logging.info('New tolerance : {0}'.format(self.m.tol))
            if self.m.tol >= kwargs['max_tol_increase']:
                logging.warning(
                'New tolerance to large for required precision'
                )
            else:
                self.m.migrad(
                    ncall = kwargs['ncall'])#, 
                    #precision = kwargs['precision']
                    #)
                logging.info(
                    'Migrad status after second try: {0}'.format(
                        self.m.migrad_ok()
                        )
                    )
                self.m.tol = tol
        return

    @setDefault(passed_kwargs = minuit_def)
    def fit(self,int_spec, minos = 0., refit = True, **kwargs):
        """
        Fit an intrinsic spectrum

        Parameters
        ----------
        intr_spec: function pointer
            function pointer to intrinsic spectrum that accepts energies in GeV and has the 
            call signature f(EGeV, **parameters)

        kwargs
        ------
        pinit: dict
            initial guess for intrinsic spectral parameters

        fix: dict
            booleans for freezing parameters

        bounds: dict
            dict with list for each parameter with min and max value
            

        Returns
        -------
        tuple with likelihood profile for distance of 
        gamma-ray emitting region
        """
        self._int_spec = lambda EGeV, **kwargs: int_spec(EGeV, **kwargs)

        fitarg = self.fill_fitarg(**kwargs)

        t1 = time.time()
        self.run_migrad(fitarg, **kwargs)

        try:
            self.m.hesse()
            logging.debug("Hesse matrix calculation finished")
        except RuntimeError as e:
            logging.warning(
            "*** Hesse matrix calculation failed: {0}".format(e)
            )

        logging.debug(self.m.fval)
        self.__repeat_migrad(**kwargs)
        logging.debug(self.m.fval)

        fmin = self.m.get_fmin()
	

        if not fmin.hesse_failed:
            try:
                self.corr = self.m.np_matrix(correlation=True)
            except:
                self.corr = -1
	
        logging.debug(self.m.values)

        if self.m.migrad_ok():
            # get the likelihood profile for the 
            # distance 
            #r, llh, ok = self.m.mnprofile(vname='r',
                #bound =kwargs['scan_bound'], 
                #bins = kwargs['r_steps'], 
                #subtract_min = False 
                #)
            r, llh, bf, ok = self.llhscan('r',
                    bounds = kwargs['scan_bound'], 
                    steps = kwargs['r_steps'], 
                    log = False
                    )
            self.m.fitarg['fix_r'] = False

            if np.min(llh) < self.m.fval and refit:
                idx = np.argmin(llh)
                if ok[idx]:
                    logging.warning("New minimum found in llh scan!")
                    fitarg = deepcopy(self.m.fitarg)
                    for k in self.parnames:
                        fitarg[k] = bf[idx][k]
                    fitarg['fix_r'] = True
                    kwargs['scipy'] = False
                    print fitarg
                    self.run_migrad(fitarg, **kwargs)

            if minos:
                for k in self.m.values.keys():
                    if kwargs['fix'][k]:
                        continue
                    self.m.minos(k,minos)
                logging.debug("Minos finished")

        else:
            self.__print_failed_fit()
            return -1,-1

        logging.info('fit took: {0}s'.format(time.time() - t1))
        for k in self.m.values.keys():
            if kwargs['fix'][k]:
                err = np.nan
            else:
                err = self.m.errors[k]
            logging.info('best fit {0:s}: {1:.5e} +/- {2:.5e}'.format(k,self.m.values[k],err))

        return r, llh 

    def llhscan(self, parname, bounds, steps, log = False):
        """
        Perform a manual scan of the likelihood for one parameter
        (inspired by mnprofile)

        Parameters
        ----------
        parname: str
            parameter that is scanned

        bounds: list or tuple
            scan bounds for parameter

        steps: int
            number of scanning steps

        {options}

        log: bool
            if true, use logarithmic scale

        Returns
        -------
        tuple of 4 lists containing the scan values, likelihood values, 
        best fit values at each scanning step, migrad_ok status
        """
        llh, pars, ok = [],[],[]
        if log:
            values = np.logscape(np.log10(bounds[0]),np.log10(bounds[1]), steps)
        else:
            values = np.linspace(bounds[0], bounds[1], steps)

        for i,v in enumerate(values):
            fitarg = deepcopy(self.m.fitarg)
            fitarg[parname] = v 
            fitarg['fix_{0:s}'.format(parname)] = True

            string_args = ", ".join(self.parnames)
            global f # needs to be global for eval to find it
            f = lambda *args: self.__calcLikelihood(*args)

            cmd_string = "lambda %s: f(%s)" % (string_args, string_args)

            minimize_f = eval(cmd_string, globals(), locals())
    
            m = minuit.Minuit(minimize_f, 
                print_level=0, forced_parameters=self.m.parameters,
                pedantic=False, **fitarg)
            m.migrad()
            llh.append(m.fval)
            pars.append(m.values)
            ok.append(m.migrad_ok())

        return values, np.array(llh), pars, ok
