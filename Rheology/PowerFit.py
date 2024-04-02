#!/usr/bin/env python
""" In order to gain some more insight to the structure of the diffusion
    process, one can use power functions (a*x**b) to fit parts of the
    MSD.
    Here this fitting procedure is implemented, which provides the
    fits and an interpolated (smoothed) MSD based on the fits.

    Author:     Tomio
    Date:       2011 June - 2012 February
    Warranty:   None
    License:    LGPL3
"""

from scipy.optimize import leastsq
from scipy.special import gamma

from numpy import *
#from Rheology import *

from matplotlib import pyplot as pl

####################################################################
# Functions to export:
__all__=[ "f_Kelvin", "f_pow",\
        "MSD_power_fit", "MSD_interpolate",\
        "fit_Kelvin", "fit_pow", "fit_pow_weight", \
        "J_to_G_Mason"]

####################################################################

#from pickle import load, dump
#from glob import glob

#get a data set:
#lst = glob('MSD-with-overlap-*.dat')
#lst.sort()
#fp = file(lst[0],'rb')
#ms = load(fp)
#fp.close()

def f_Kelvin(x, a, b):
    """ Simple Kelvin body like approximation:

        return:
        a*(1-exp(-b*x))

        a and b are simple floating point constants
    """
    if len(a) != len(b):
        print("Mismatch in parameters length")
        return None

    y = zeros(x.shape)

    for (i,j) in zip(a,b):
        y = y + i*(1.0-exp(-j*x))

    return y
#end of f_Kelvin

def f_pow(x, a, b):
    """ Calculate the sum of power functions:
        f = sum for i : a[i]*x**b[i]

        a,b are lists (or vectors) with the same length

        if x is a numarray vector the return value will be an
        array with the same dimensions

        return: f
    """
    if len(a) != len(b):
        print("Mismatch in parameters")
        return None

    if type(x).__name__ != 'ndarray':
        y = 0.0
    else:
        y = zeros(x.shape)
    #end if

    y = y
    for (i,j) in zip(a,b):
        y = y + i*x**j
    #end for

    return y
#end f_pow

def f_pow2(x, a,b,c):
    """ Calculate a second order polynomial power law...
        (the crazy thing needed for the extended Mason method)

        ln(y) = a+b*ln(x)+c*ln(x)**2

        This can be calculated in log-log and turned back, or
        done in linear space using:
        y = a*(x**b)*(x**(c*ln(x)))

        return f(x)
    """
    lx = log(x)

    power = b+c*lx

    return a*x**power
#end of f_pow2

def err_pow_weight(parms, *args):
    """ chi^2 function for the leastsq().
        parameters: a, b and an optional c
        data (*args): x, y, weight

        Return:
        (yy - y)*weight
        where   yy = a*(x**b)
        or:     yy = a*x**(b + c*log(x))
        this latter is equivalent to a second order
        polynomial fit in log-log space

        leastsq does the sqare and the sum up
    """
    NA = len(args)
    N = len(parms)

    if NA != 3:
        print("Invalid parameters")
        raise ValueError

    x = args[0]
    y = args[1]
    weight = args[2]

    #we want to go down to the lowest level:
    #x**a = exp(a*log(x))

    #this log is fine:
    lx = log(x)

    if N < 2 or N > 3:
        print("Invalid parameters")
        return -1

    a = parms[0]
    power = parms[1]*lx

    if N > 2:
        c = parms[2]
        #to fit an a+b*x+c*x*x in log-log space, one gets:
        #yy = a*(x**(b+c*log(x))) = a*exp(b*log(x) + c*log(x)^2)
        #else:
        #yy = a*(x**b)
        power = power + c*lx*lx

    #Constrains:
    #exp(710) is overflow, returning inf
    #exp(100) > 10^43...
    atest = abs(a)
    #theoretically power > 1 is already fishy
    if power.max() > 10 or power.max()*log(atest) > 250 :
        #chi = 1000*y*power*a
        chi = zeros(x.shape)+1E99
    else:
        yy = a*exp(power)
        chi = (yy - y)*weight

    #power of 2:
    # the leastsq takes the square of the return
    #value, and sums it up as well
    #chi = chi*chi

    return chi
#end err_pow_weight

def err_pow(parms, *args):
    """ chi^2 function for the leastsq().
    """
    NA = len(args)
    N = len(parms)
    xx = args[0]
    y = args[1]

    if N < 2:
        print("Invalid parameters")
        return -1

    N2 = int(N/2)
    a = parms[:N2]
    b = parms[N2:]

    yy = f_pow(xx, a, b)

    chi = yy - y
    #power of 2:
    # the leastsq takes the square of the return
    #value, and sums it up as well
    #chi = chi*chi

    return chi
#enf err_pow

def err_Kelvin(parms, *args):
    """ chi^2 for the lestsq().
        parms: a and b
        args: (x,y) or [x,y]

        added constrain: a < 0: err = 1E99
    """
    N = len(parms)
    if N < 2:
        print("Wrong number of parameters! (!=2)")
        return -1

    N2 = int(N/2)
    a = parms[:N2]
    b = parms[N2:]

    xx = args[0]
    yy = args[1]
    y = f_Kelvin(xx, a, b)

    #power of 2:
    # the leastsq takes the square of the return
    #value, and sums it up as well
    #chi = chi*chi

    chi = y - yy

    return chi
#end of err_Kelvin

def fit_pow_weight(x,y, weight, parabolic=True):
    """ Fit an a*x**b function with weights,
        used as a multiplier in the error estimation.

        This one is needed for estimating the exponent in
        the conversion method of Mason et al.

        if parabolic is True, then fit a second order
        polynomial in log-log space

        return:
            a, b, chi^2(x)
    """
    if len(x) != len(y) or len(x) != len(weight):
        print("\ninvalid parameter lengths\n")
        raise ValueError

    indx = (x>0) & (y>0) & (x < inf) & (y < inf)
    #indx = (x>0) & (y>0)
    x = x[indx]
    y = y[indx]
    weight = weight[indx]

    if len(x)< 3:
        print("Invalid data!")
        return {'a':0.0, 'b':0.0, 'c':0.0,'x':x, 'y':y, 'fitted':zeros(x.shape)}

    x0 = log(x)
    y0 = log(y)

    if parabolic:
        #the coefficients come as largest,....,lowest power:
        #2, 1,  0
        c0, b0, a0 = list(polyfit(x0, y0, 2))

    else:
        b0, a0 = list(polyfit(x0, y0, 1))


    parms= [exp(a0), b0, c0] if parabolic else [exp(a0),b0]

    #leastsq:
    # error function
    # parameters to be optimized
    # further arguments to the function
    # tuning switches
    #indx = (x > 0.0)
    #indx = indx & (y > 0.0)
    #x= x[indx]
    #y= y[indx]
    #output:
    res = dict()

    try:
        fit = leastsq( err_pow_weight,\
            parms, \
            (x,y, weight),\
            full_output= True, maxfev= 1000000 )
        res['cov_x'] = fit[1]

    except (ValueError, LinAlgError):
        fit = leastsq( err_pow_weight,\
            parms, \
            (x,y, weight),\
            full_output= False, maxfev= 1000000 )
        res['cov_x'] = None


    res['x'] = x
    res['y'] = y
    res['a'] = fit[0][0]
    res['b'] = fit[0][1]

    if parabolic and len(fit[0]) == 3:
        res['c'] = fit[0][2]

    res['fitted'] = res['a']*exp(res['b']*x0) if not parabolic else\
                    res['a']*exp(res['b']*x0 + res['c']*x0*x0)

    res['err'] = ((res['fitted'] - res['y'])**2)
    res['weights'] = weight

    return res
#end fit_t

def fit_pow(x, y):
    """ Take a data set x,y, and fit it with a simple
        power function: a*x**b.

        Return:
            a, b, chi^2[x]
    """

    x0 = log(x)
    y0 = log(y)
    b0, a0 = list(polyfit(x0, y0, 1))
    parms = [exp(a0),b0]

    #leastsq:
    # error function
    # parameters to be optimized
    # further arguments to the function
    # tuning switches
    fit = leastsq(err_pow,\
            parms, \
            (x,y),\
            full_output= True, maxfev= 1000000 )

    res = dict()
    res['x'] = x
    res['y'] = y
    res['a'] = fit[0][:1]
    res['b'] = fit[0][1:]
    res['fitted'] = f_pow(res['x'],res['a'],res['b'])
    res['err'] = ((res['fitted']-res['y'])**2)
    res['cov_x'] = fit[1]
    res['type'] = "power"
    #print("Fitted a power law model with parameters", res['a'], res['b'])

    return res
#end fit pow


def fit_Kelvin(x, y, N=1):
    """ Take a data set x,y, and fit it with a simple
        Kelvin solid function: a*(1-exp(-b*x))
        Parameters:
        x:      x data (independent variable)
        y:      y data (dependent variable, to be fitted)
        N:      how many parameters to use

        Return:
            a, b, chi^2[x]
    """
    x = x.ravel()
    y = y.ravel()

    if N < 1:
        print("Invalid range %d, forcing 1" %N)
        N = 1
    #N tells how many segments one wants to fit
    #st is the length of each
    st = int(y.size/N)
    #the curve must be saturating, so each end value is
    #a good start for a
    a0 = y[st*arange(N)-1]
    b0 = zeros(N, dtype=float)

    #how do you estimate the exponent?
    for i in range(N):
        i0 = i*st
        i1 = min(i0+1,y.size)
        b0[i] = abs((y[i0]-y[i1])/(a0[i]*(x[i0]-x[i1]))) \
                        if a0[i] != 0.0 and x[i0]!=x[i1] else 10.0

    print("Startin K-V parameters", a0, b0)
    parms = concatenate((a0,b0))

    #leastsq:
    # error function
    # parameters to be optimized
    # further arguments to the function
    # tuning switches
    fit = leastsq(err_Kelvin,\
            parms, \
            (x,y),\
            full_output= True, maxfev= 1000000 )

    res = dict()
    res['x'] = x
    res['y'] = y
    res['a'] = fit[0][:N]
    res['b'] = fit[0][N:]
    print("Fitted a sum of %d Kelvin/Voigt models" %N)
    print("Parameters:")
    print(res['a'])
    print(res['b'])
    res['fitted'] = f_Kelvin(res['x'],res['a'],res['b'])
    res['err'] = ((res['fitted']-res['y'])**2)
    res['cov_x'] = fit[1]
    res['type'] = "Kelvin"

    return res
#end of fit_Kelvin

def Kelvin_power_cross(a0,b0, a1,b1, t0):
    """ To estimate where are the Kelvin-Voigt and a
        power law MSDs are the closest, we need nonlinear
        fitting again. The equation otherwise is just not
        simple to solve.

        a0,b0: the Kelvin-Voigt profile is a0*(1-exp(-b0 x)
        a1,b1: the power law is a1*(x**b1)
        t0:     start searching here

        return: the x value found
    """
    if a0 <= 0.0 or a1 <= 0.0:
        print("\t\tWARNING:, invalid Kelvin parameter!\n")
#do not crash on it
#        return 0.0

    def f(x):
        return abs(a1*(x**b1) - a0*(1.0-exp(-1.0*b0*x)) )

    fit = leastsq(f,\
            t0, \
            full_output= True, maxfev= 1000000 )
    return fit
#end Kelvin_power_cross


def MSD_power_fit(ms, t0, fill=0.9, scaler=10,\
                Kelvin=False, ReEval=False,\
                mode='scaler',\
                verbose=False):
    """ Interpolate the MSD using power functions and nonlinear fitting.
        This interpolation is based on the assumption, that the MSD of
        whatever diffusion is a power law in certain time regimes.
        This algorithm tries fitting power function to various parts of
        the MSD. It checks the fit error and adjusts the intervals to
        minimize these errors.

        When complete, the crossing of the interpolations is checked,
        and if they are within the time window, the intevrvals are adjusted
        to have the crossing at the end between.

        Parameters:
        ms:         an MSD, MKD or J dict
        t0:         the first fit is within 0:t0
        fill:       up to what rate (0-1) use the data. This can be used to cut
                    the end  of the data, due to noise or other reasons.
        scaler:     this is used to estimate a starting upper limit of fitting
                    i1= scaler*i0

        Kelvin:     If True, the first data set is fitted with f_Kelvin()
        ReEval:     re-evaluate a fit if the boundaries changed?
                    During testing for crossing points, the index range of a
                    fit changes. This may affect the fitted values, which may
                    result in a new crossing point etc.
                    Nothing ensures convergence, so we do not iterate with this.
                    Re-evaluation is done only once.

        mode:       a string defining the mode how the initial fit intervals
                    are estimated. Possible values are:
                    'scaler':   i1 = scaler*i0 (default)
                    'interval': the start interval scales: di = di*scaler
                    'delayed':  if KelvinStart, then the next first interval
                                uses the resulted interval of the Kelvin fit
                                then starts using scaler as above
                    'inherit':  inherit the final interval of the previous fit
                    'inheritscaler':
                                take the interval of the previous fit and
                                multiply it with scaler


        verbose:    plot the results

        Return:
        a list of dicts, each dict containing fitting information
        'x':    time data fitted
        'y':    MSD values fitted
        'a':    a from a*x**b
        'b':    b from a*x**b
        'fitted':   fitted data points
        'err':      error square of the points
        'cov_x':    the covariance matrix provided by leastsq
        'i0','i1':  ms[..][i0:i1] is used
        't0', 't1': t0 <= ms['tau'] < t1
        'type':     Kelvin or power (which function was used)
    """
    #for keyword check:
    modes =['scaler','interval','delayed', 'inherit', 'inheritscaler']

    if fill < 0.1 or fill > 1.0:
        print("Invalid fill parameter %.3f, resetting to 0.9" %fill)
        fill = 0.9
    if scaler <= 1.0 :
        print("Invalid scaler %.1f, resetting to 10" %scaler)
        scaler = 10
    if t0 > max(ms['tau']):
        print("Invalid t0: %f" %t0)
        return []
    if mode not in modes:
        print("Invalid scaler mode %s. Falling back to 'scaler" %mode)
        mode = 'scaler'
    #end if

    if ms.has_key('MSD'):
        y = ms['MSD']
    elif ms.has_key('MKD'):
        y = ms['MKD']
    elif ms.has_key('J'):
        y = ms['J']
    else:
        print("unknown data set")
        return []
    #end if

    if ms.has_key('tau'):
        x = ms['tau']
    else:
        print("tau not found error")
        return []

    #for verbose output, plot the input first
    #both on linear and log scale
    if verbose:
        fig1 = pl.figure(1)
        pl.clf()
        plt1 = fig1.add_subplot(1,1,1)
        pl.xlabel("$\\tau$, seconds")
        pl.ylabel("Y data")
        plt1.loglog(x,y,'o')

        fig2 = pl.figure(2)
        pl.clf()
        plt2 = fig2.add_subplot(1,1,1)
        pl.xlabel("$\\tau$, seconds")
        pl.ylabel("Y data")
        plt2.plot(x,y,'o')
        #plt2.axis((0,max(x),0, max(y)))

    fits = []
    NFM = 5 #number of minimum fit points
    N = len(x)
    i0 = 0
    #In general: we should fit more than 3 points!
    i1 = min( max(NFM, sum(x < t0)), N)
    #we need the step size later
    #to use for interval scaler:
    di = i1 - i0
    #we want to fit > NFM points:
    Nend = min( int(fill*float(N)), N-NFM)

    #make the fitting function adjustable:
    fitF = fit_Kelvin if Kelvin else fit_pow
    errF = err_Kelvin if Kelvin else err_pow
    fF   = f_Kelvin if Kelvin else f_pow
    #switch: this is the first fit,
    #which matter only for Kelvin
    sw = True if Kelvin else False

    #Now, the real work:

    #loop through the data set
    while i0 < Nend:

        if verbose:
            print("Actual i0 (from %d): %d" %(Nend, i0))

        #only the first part is fitted with K/V model
        #switch back to the power fit:
        if sw and i0 > 0:
            fitF = fit_pow
            errF = err_pow
            fF = f_pow
            sw = False
        #end if sw

        #do a fit to start with:
        fit = fitF(x[i0:i1],y[i0:i1])
        err = errF((fit['a'],fit['b']),x,y)**2
        err -= mean(err[i0:i1])
        #the negative values of err are definitely better errors
        #than the mean. The last one of them is an inferior estimate
        #of the upper index
        #now we force it into the original fit range
        #with this we do not allow jump over non-fitting areas
        #outside the pre-defined range
        errmin = nonzero(err[:i1] < 0)[0]

        #this should not happen...
        if len(errmin) < 1:
            raise ValueError("Error: no errors below mean? (%d)" %len(err))
        #end if

        #now, we need the last possible hit
        #what to do if this comes plenty later, but not continuously?

        j1 = max(errmin)

        if j1 < i0:
            print("we have a problem, invalid fit!")
            #start over
            i0 = j1
        else:
            #where do we deviate from the fit first, but above the
            #minimum fit index
            #this is a better estimate of the upper index
            j2 = nonzero(err[j1:] > std(err[i0:j1]))[0]
            j2 = min(j2) + j1 + 1 if len(j2) > 0 else j1 + 1

            #refine the fit:
            i1 = min(j2, N)
            fit = fitF(x[i0:i1],y[i0:i1])
            #archive the achieved fit range:
            fit['t0'] = x[i0]
            fit['t1'] = x[min(j2,N-1)]
            fit['i0'] = i0
            fit['i1'] = i1
            fits.append(fit)

            #define the next range:
            if mode == 'scaler':
                i0 = j2
                di = int( (scaler-1.0)*i0 )
                i1 = i0 + max(di, NFM)

            elif mode == 'interval':
                di = int(di * scaler)
                i0 = j2
                i1 = i0 + max(di, NFM)
            elif mode == 'delayed':
                di = di if sw else max(j2 - i0, NFM)
                i0 = j2
                i1 = i0 + di if sw else int( i0*scaler )

            elif mode == 'inherit':
                di = j2 - i0
                i0 = j2
                i1 = i0 + di

            elif mode == 'inheritscaler':
                di = max((j2 - i0)*scaler, NFM)
                i0 = j2
                i1 = i0 + di
            else:
                print("ERROR: Invalid scaler mode!")
                return []
            #end if mode

            if verbose:
                #in a range, the last index is not reached!
                print("Fit range: %d - %d"  %(i0,i1))
                if i0 < N and i1 <N:
                    print("\t\t %.3f - %.3f" %(x[i0],x[i1]))

                plt1.loglog(x, fF(x,fit['a'],fit['b']),'-')
                plt2.plot(x, fF(x,fit['a'],fit['b']),'-')
        #end if j1 < i0
    #end of while

    if verbose:
        pl.draw()

    #clean up:
    # do the fits cross?

    sw= True if Kelvin else False
    print("Checking for crossing points")
    for i in range(1, len(fits)):
        #do the two log-log fits cross somewhere?
        #t0 holds the estimated crossing point (if any):
        if sw:
            a0 = fits[i-1]['a']
            b0 = fits[i-1]['b']
            a1 = fits[i]['a']
            b1 = fits[i]['b']

            #t0 holds the crossing point:
            t0 = Kelvin_power_cross(a0,b0, a1,b1, fits[i-1]['t1'])[0][0]

            #set the first function for re-evaulation (if needed)
            fitF = fit_Kelvin
            errF = err_Kelvin
            fF   = f_Kelvin
            #only first time to be used, no more:
            sw = False
        else:
            #set the first function for re-evaulation (if needed)
            fitF = fit_pow
            errF = err_pow
            fF   = f_pow

            a1 = log(fits[i]['a'])
            a0 = log(fits[i-1]['a'])
            b1 = fits[i]['b']
            b0 = fits[i-1]['b']
            #if b0 == b1: they never meet...
            #one can also use in general:
            #abs(f1-f2).min() to find the
            #closest point or zero...
            if b0 != b1:
                lgx = (a1-a0)/(b0-b1)
                print(lgx)
                t0 = exp(lgx)

                print("Found t0: %.5f" %t0)
        #end if sw: we have t0

        #is it within the fitting range?
        #then we should redefine their ranges:
        if t0 > fits[i-1]['t0'] and t0 < fits[i]['t1']:
            print("it is a new matching point")
            #taking the right side of the match:
            i0 = sum( x < t0)
            #this should not happen:
            if i0 > N-2:
                print("i0 is out of range!")
                continue

            if (i0 - fits[i-1]['i0']) < NFM or \
                (fits[i]['i1'] - i0) < NFM:
                print("New limit would erase a fit! Skip it!")
                continue

            print("i: ", i0, "t:", x[i0])

            #redefine the fits:
            print("redefining fit %d" %(i-1))
            of = fits[i-1]
            #re-evaluation: this may lead away from the original!
            #the first function depends on Kelvin
            if ReEval:
                fits[i-1] = fitF(x[of['i0']:i0],y[of['i0']:i0])
            #keep the old left side:
            fits[i-1]['t0'] = of['t0']
            fits[i-1]['i0'] = of['i0']
            #redefine the right side:
            fits[i-1]['t1'] = x[i0]
            fits[i-1]['i1'] = i0

            print("redefining fit %d" %(i))
            of = fits[i]
            #the second function is always power function:
            if ReEval:
                fits[i] = fit_pow(x[i0:of['i1']],y[i0:of['i1']])
            #redefine the left side:
            fits[i]['t0'] = x[i0]
            fits[i]['i0'] = i0
            #keep the right side
            fits[i]['t1'] = of['t1']
            fits[i]['i1'] = of['i1']
    return fits
#end of MSD_power_fit

def dynamic_interpolation(x, a, b, IsKelvin=True, eps=1E-4,\
        insert=10.0, verbose= True):
    """ simple dynamic interpolation using Kelvin body
        or power law.

        x: data array
        a,b: if Kelvin model then a*(1-exp(-b*x))
                else power law of a*x^b is used
        IsKelvin:   use the Kelvin model else the power law
        eps:        what precision (actually sqareroot of it)
        insert:     insert this many new points between t0 and 0

        returns a resampled x, y set

    """
    if eps <= 0.0:
        eps = 1E-6

    if verbose:
        print("a: %.3f, b: %.3f" %(a,b))

    if insert >= 1.0:
        x0 = x[0]/float(insert)
        xx = concatenate(( arange(x0,x[0]-x0, x0),x))
    else:
        xx = x.copy()
    #offset is the shift between i in the original x
    #and the new x data set:
    offset = 0
    #the result array is first identical to the old one:
    xnew = xx.copy()

    for i in range(1, xx.size):
        xi = xx[i]
        x0 = xx[i-1]

        if xi <= 0.0:
            print("I can not handle x <= 0")
            continue;

        if IsKelvin:
            h0 = eps*sqrt((exp(b*x0)-1.0))/b
            h1 = eps*sqrt((exp(b*xi)-1.0))/b
        else:
            h0 = eps*x0/sqrt(abs(b*(b-1)))
            h1 = eps*xi/sqrt(abs(b*(b-1)))

        hi = min(h0, h1)
        #print("New interval: %.5f instead of %.5f" %(hi, (xi - x0)))

        if hi < (xi - x0):
            #resample if there is refinement:
            #to insert equally spaced, N number of points,
            #we have to round up N and then calculate hs.
            #The resulted accuracy is supremum of the desired one
            iN = int( (xi-x0)/hi +1.0)
            hs = (xi-x0)/float(iN)
            #new data points between x0 and xi,
            #excluding xi (we keep the original)
            xinsert = arange(x0+hs, xi-hs/10.0, hs)
            i0 = i + offset

            #xnew = xtmp
            xnew = concatenate( (xnew[:i0], xinsert, xnew[i0:]))
            offset += xinsert.size
            #print("refining is done, %d points" %(xnew.size))
    #end for xi

    if verbose:
        print("The new array has %d points" %xnew.size)

    return xnew
#end of dynamic_interpolation

def MSD_interpolate(ms, fitsin, smoothing=True,\
                    Nsmoothing=30,\
                    insert= 2,\
                    factor= 3.0,\
                    eps= 1E-4,\
                    verbose=False):
    """
        Interpolate the data of the MSD, based on the fit dictionary generated
        by MSD_power_interpolation.
        Parameters:
        ms:             an MSD, MKD or a J dict
        fitsin:         a list of fit dicts from MSD_power_interpolation
        smoothing:      apply an averaging around the transition point
                        within the fit region
        Nsmoothing:     the desired width of the running average in data points
                        is used.
                        (y[j0:j1] = s*gs + (1.0-s)*fs, where gs and fs are
                        the fitting functions in the two intervals;
                        j0 = i- Nsmoothing/2; j1 = i+Nsmoothing/2; i is the
                        crossing point)

        insert:         insert this many data points between 0 and tau[0]
        factor:         oversampling the original time array
                        Take the original time window (min and max),
                        and resample the data to factor*N data points
                        (include the inserted ones in N)

                        if -1, then turn on the dynamic resampling:
                        this uses the second derivative and the function
                        to estimate a dynamic step size

        eps:            an accuracy factor for dynamic interpolation
                        (below 1E-4 the array may grow a lot!)

        verbose:        plot things and talk a bit

        return:
            a new MSD dict containing the interpolated MSD values.
            dtau and DMSD are erased.

        The data is truncated to the last end index in fits.
        (fits[-1]['i1'])
    """

    #interpolation:
    #this is not the bests, but are there better ways?
    ms2 = ms.copy()
    fits = list(fitsin)

    if ms.has_key('MSD'):
        y = ms2['MSD']
        yold = ms['MSD']
    elif ms.has_key('MKD'):
        y = ms2['MKD']
        yold = ms['MKD']
    elif ms.has_key('J'):
        y = ms2['J']
        yold = ms['J']
    else:
        print("unknown dict!")
        return {}

    if ms2.has_key('tau'):
        xold = ms['tau']
        x = ms2['tau']
    else:
        print("Unknown structure, tau (x-data) not found!")
        return {}


    #dynamic resampling can also insert, so we do it here
    #if factor is not -1:
    if insert > 0 and factor != -1:
        ddx = x[0]/float(insert)
        dx = arange(ddx, x[0]-ddx/10.0, ddx)
        x = concatenate((dx,x))
    #end if insert

    #save the original boundary points before we change them:
    orig= []
    for i in fits:
        orig.append( [i['i0'], i['i1']] )
    #end creating orig

    if factor > 1.0 or factor == -1.0:
        if factor == -1:
            newx = asarray([], dtype=float)
            newi0 = 0
            N = len(x)

            for i in fits:
                #the stored positions in the original x:
                i0 = i['i0']
                #we need to send the end points as well!
                i1 = min(i['i1']+1, N)
                #fit parameters:
                a = i['a']
                b = i['b']
                tp = (i['type'].lower() == 'kelvin')

                if i0 == 0:
                    nx = dynamic_interpolation(x[i0:i1], a, b,\
                            IsKelvin=tp, eps=eps, insert= insert, \
                            verbose= verbose)
                    newx = concatenate( (newx, nx))
                    newN = len(nx)
                else:
                    nx = dynamic_interpolation(x[i0:i1], a, b,\
                            IsKelvin=tp, eps=eps, insert= 0, \
                            verbose= verbose)
                    #the first point in nx is the same as the last
                    #in newx. So, drop it:
                    newx = concatenate( (newx, nx[1:]))
                    newN = len(nx) -1
                #end if i0
                #now we have a new chunk,
                #in the new array, this is newi0:(newi0+len(nx))
                #we have to store this:

                #reposition:
                i['i0'] = newi0
                newi0= i['i1'] = newi0 + newN

        else:
            #the new x array for the oversampling:
            newx = arange(x.min(),x.max(), (x.max()-x.min())/(x.size*factor))

            #adjust boundary indices:
            for i in fits:
#               print("old i0:i1: %d:%d (%.4f:%.4f)" %(i['i0'],i['i1'], \
#                                                   i['t0'],i['t1']))
                i['i0'] = (newx < i['t0']).sum()
                i['i1'] = (newx < i['t1']).sum()
#               print("new i0:i1: %d:%d (%.4f:%.4f)" %(i['i0'],i['i1'], \
#                               newx[i['i0']], newx[i['i1']]))
            #end for i in fits
        x = newx
        y = zeros(x.shape, dtype=float)

    #Calculate the interpolated values:
    for i in fits:
        i0 = i['i0']
        i1 = i['i1']
        tp = (i['type'].lower() == "kelvin")

        print(i0, i1)
        y[i0:i1] = f_Kelvin(x[i0:i1], i['a'],i['b']) if tp else\
                             f_pow(x[i0:i1], i['a'],i['b'])
    #end for
#    print("New array has size: (X) %d, (Y) %d" %(len(x),len(y)))
#    print("%d NaNs in the data" %(sum(y == NaN)))
    N = len(y)
    Nold = len(xold)

    #should we smooth the transient region?
    #If so, use a simple running average in the range of Nsmoothing
    # around i0s.
    if smoothing:
        #define smoothing according to the original not-interpolated data set!
        if Nsmoothing < 0 or Nsmoothing > len(yold):
            if verbose:
                print("invalid smoothing region: %d" %Nsmoothing)
                print("switching to dynamic")
            Nsmoothing = 0
        #end if Nsmoothing
        #for i in range(1, len(fits)):
        for i in range(1, len(orig)):
            #we smooth between 2 fits: fits[i-1] and fits[i]
            #how many points can we run through?

            #If not specified, then maximum the 1/4 of the data range
            #in both (+/-) direction
            if Nsmoothing == 0:
                Nsmoothing = min( (orig[i][1]-orig[i][0])/4,\
                                 (orig[i-1][1]-orig[i-1][0])/4)
            #end if

            Ns2 = int(Nsmoothing/2)
            #Our meeting point is i0 in the original:
            i0 = orig[i][0]
            #limits: the previous 0 and the nex i1
            ip0 = orig[i-1][0]
            in1 = orig[i][1]

            #we run around this, but do we ever run out?
            #and transform to the interpolated index!
            j0 = (x <= xold[max(ip0, i0 - Ns2)]).sum()
            j1 = (x <= xold[min(in1, i0 + Ns2)]).sum()

            if verbose:
                print("Smoothing around %d" %i0)

                if (j0 == 0 or j1 == N or j0==j1):
                    print("Warning! Nsmooth too large,\
                            cutting boundaries to: %d:%d" %(j0,j1))

            #one can use a tanh to switch between the two functions
            #the smoothing is done around i0 in size Nsmoothing
            #there are various possibilities, such as a tanh() as well
            #for tanh() xs has to go from -0.5 to 0.5
            #xs = -0.5 ... 0.5
            #s = 0.5+ 0.5*tanh(7.0*xs)
            #take the linear interpolation between the two functions:
            #s = 0...1 between the two ends
            s = arange(j1-j0, dtype=float)/float(j1-j0)
            #print(s.min(), s.max(), s.mean())

            #fs is the first part, gs is the second
            #smoothing is: at the beginning fs (s=0), at the end gs (s=1)
            fs = f_Kelvin(x[j0:j1],fits[i-1]['a'],fits[i-1]['b']) \
                    if fits[i-1]['type'].lower() == "kelvin"  else \
                        f_pow(x[j0:j1], fits[i-1]['a'],fits[i-1]['b'])

            gs = f_Kelvin(x[j0:j1],fits[i]['a'],fits[i]['b']) \
                    if fits[i]['type'].lower() == "kelvin"  else \
                        f_pow(x[j0:j1], fits[i]['a'],fits[i]['b'])

            #the transition is from gs to fs
            #interpolate and replace:
            y[j0:j1] = s*gs + (1.0-s)*fs
        #end for i

    imax = fits[-1]['i1']

    if ms2.has_key('dtau'):
        ms2.pop('dtau')
    if ms2.has_key('DMSD'):
        ms2.pop('DMSD')

    ms2['tau'] = x.copy()
    ms2['MSD'] = y
    #truncate to the interpolated range:
    for i in ms2.keys():
        ms2[i] = ms2[i][:imax]

    if verbose:
        fig = pl.figure()
        plt = fig.add_subplot(1,1,1)
        plt.plot(xold,yold,'b+')
        plt.plot(x,y,'r-')
    pl.draw()

    return ms2
#end MSD_interpolate

def J_to_G_Mason(J, N=30, \
                r0= 5, w= 1.0, \
                logstep=True, omax=0.0,\
                logWeight=True, advanced=True,\
                verbose=False, flip=True):
    """ Convert creep compliance dict to G using the method published
        by Mason's group (Dasgupta et al. PRE 65:051505 (2002)).

        This algorithm does not use any correction factors.

        The method fits the creep-compliance with a power function
        locally, then uses the fitted a*t**b to calculate G* locally
        as:
        G(1/t) = exp(i pi/2 b) (1/t)**b / (a * gamma(1+b))

        The local fit is weighted with a gaussian of width w:
        exp( - (t-t0)/(2*width**2))

        Parameters:
        J:          a dict from MSD_to_J
        N:          the number of points to take
        r0:         2r0+1 points are used in the fit
        w:          w parameter for the Gaussian weights
        logstep:    make a logarithmic array or else a linear
        omax:       maximal omega value, if <= 0 or > 1/dt
                    then use 1/dt.
        logWeight:  We use weights around the given point to calculate
                    the fit. These weights follow a Gaussian, but is the
                    Gaussian form in linear space or in the log-log space?
                    If True, then log-log.
                    The weights are generated according to index, not the time!
                    (this may be problematic for not equally spaced data, but
                    eliminates other problem with selecting the width parameter
                    for the Gaussian properly)

        advanced:   there is an "improved" version of the conversion published
                    in the above paper, which uses a parabolic fit in log-log
                    space, then again fits the converted G to improve the phase
                    information.
                    If True, use this algorithm, else the simple power fit and
                    direct conversion.

        verbose:    talk back a bit
        flip:       flip the arrays around, since the default is
                    resulting in decreasing f and omega

        Return:
        a dict containing:
        'f':        frequency values
        'omega':    circular frequency values
        'G':        complex values of G
    """

    if r0 < 1:
        r0 = 1.0

    if w <= 0.0 :
        w = 1.0

    if not J.has_key('tau') or not J.has_key('J'):
        raise ValueError("Critical parts of J are missing!")
    #end ifs...

    #generate the Gaussian weight distribution:
    if logWeight:
        weight = log(arange(1, 2*r0+2, dtype='f')/r0) /w
        #taking a log(x) log(y) plot, this will be a Gaussian:
        weight = exp(exp(-0.5*weight**2))

    else:
        weight = arange((2*r0+1), dtype='f')
        weight = exp(-(weight - r0)**2/(2* w*w) )
    #normalize to 1 (not necessary, but can be nice)
    weight = weight/weight.max()

    Nw = len(weight)

    if verbose:
        print("\nWeight array:", weight)
        fig = pl.figure(1)
        fig.clf()
        plt = fig.add_subplot(111)

    #as a time step we take the smallest nonzero time point:
    dt = (J['tau'][J['tau']>0.0]).min()
    #we could also use:
    #dt = J['tau']; dt = (dt[1:] - dt[:-1]).mean()

    #the requested omega max defines a minimum
    #time interval as well
    dtmin = 1.0/omax if omax > 0.0 and omax <= 1.0/dt else dt
    #for this "fit" the omax = 1/dt, thus we use data starting
    #at tau= dtmin:
    i0 = (J['tau'] < dtmin).sum()
    NJ = len(J['tau'][i0:])

    #if we do advanced, then we need more points than proposed:
    if advanced:
        N = N + int(2*r0+1)

    #we take a subsets of the original data to
    #estimate what frequency values we shall have...
    #These points are the core points around which we do the fitting
    # and calculate G*(omega)
    if logstep:
        #it would be exp( arange(0,log(NJ) ),log(NJ)/N), but this is a
        #less tedious way:
        indx= exp(log(float(NJ))*arange(N, dtype='f')/float(N))
    else:
        #resample the NJ points to N indices:
        indx = float(NJ)*arange(N, dtype='f')/float(N)

    #we need the indices as integers, so round them:
    indx = (indx+0.5).astype(int)
    #this may result multiple integers in the index list
    #now we have to clean those out:
    #first keep the last index as an array:
    lastindx = indx[-1:]
    #we keep indices, where the index change is > 0:
    indx = indx[ (indx[1:]-indx[:-1]).nonzero() ]

    #start on the data:
    x= J['tau']
    y= J['J']
    #the last point we should not lose
    #and multiple indices occur at the low index end
    if lastindx != indx[-1]:
        indx = concatenate((indx,lastindx))

    #We work with logs: negative vaules and infinity are not
    #good!
    filtindx = (x>0)&(x<inf) &(y>0)&(y<inf)
    if x.size != filtindx.sum() :
        print("Found %d invalid data points!" %(x.size - filtindx.sum()))

    x = x[filtindx]
    y = y[filtindx]

    omega = []
    G = []
    dG = []

    cip = complex(0.0, pi/2.0)
    #indx was created from [i0:], so shift to i0:
    indx = indx + i0

    alphalist = []
    glist=[]
    dGlist= []
    #fit around the selected points of the J, and
    #calculate the G(omega) set.

    print("entering first phase")
    for i in indx:
        i0 = int(max(i-r0, 0))
        i1 = int(min(i+r0+1, NJ))
        j0 = 0 if i > r0 else r0-i
        j1 = Nw if i < (NJ-r0) else NJ-i

        #generate the local power law fit: a*t^b
        fit= fit_pow_weight(x[i0:i1],y[i0:i1], weight[j0:j1],\
                            parabolic= advanced)
        #a is the magnitude (scale) of J -> divide with it
        #b is the exponent
        #alpha = b for the linear case,
        #alpha = dlog(y)/dlog(x)=b + 2*c*log(x[i]) for the parabolic case
        #c is the nonlinear part, beta in the article
        a = fit['a']
        beta = 2.0*fit['c'] if advanced else 0.0
        alpha = fit['b'] + 2.0*fit['c']*log(x[i]) if advanced else fit['b']

        om = 1.0/x[i]

        #this G is still real:
        #the position is shifted in the small array vs. the original:
        #print(fit['fitted'])
        #print("parms", fit['a'], fit['b'], fit['c'])
        yi = fit['fitted'][i - i0]
        #This is the general symmetry of power function for Fourier
        #transform. yi is J(t=1/omega)
        g = 1.0/(yi*gamma(1.0+alpha))
        #using the direct Fourier transfrom of an power function:
        #g = om**alpha/(a*gamma(1.0+alpha))

        if advanced:
            g = g/(1.0+0.5*beta)

        if g < 0 or g == inf :
            print("Invalid G* dropped")
            #skipping
            continue

        if verbose:
            plt.cla()
            plt.plot(fit['x'],fit['y'],'bo')
            plt.plot(fit['x'],fit['fitted'],'r-+')
            pl.draw()

        #the result is g*alpha for the simple case
        #alpha carries the complex part:
        alphalist.append( alpha )

        omega.append(om)
        glist.append(g)
        dGlist.append(fit['err'].sum())
    #end for i

    NO = len(omega)

    #lock this out for now:
    if advanced and (NO-2*r0) > 2:
        omega= asarray(omega)
        NO = len(omega)
        gs = asarray(glist)
        pi2 = pi/2
        pi21 = pi2 - 1.0

        glist = []
        omlist = []
        dG= []

        print("second phase")
        r0 = int(r0)

        for i in arange(r0, NO-r0-1):
            #refit the converted data (now it is the parabolic case!)
            fit= fit_pow_weight(omega[i-r0:i+r0+1],gs[i-r0:i+r0+1], \
                    weight, True)

            #rule is the same as before for alpha and beta:
            a = fit['a'];
            beta = fit['c']
            alpha = fit['b'] + 2.0*fit['c']*log(omega[i])

            #calculate the phase part using a,b,c:
            pre = 1.0/(1.0+beta)
            alphapi = pi2*alpha
            #phase part of equations 5 and 6:
            alphaR = cos(alphapi - pi21*alpha*beta)
            alphaI = sin(alphapi- pi21*beta*(1.0-alpha))

            #combine to the complex shear modulus:
            g = gs[i]*pre*complex(alphaR, alphaI)

            if verbose:
                plt.cla()
                plt.plot(fit['x'],fit['y'],'bo')
                plt.plot(fit['x'],fit['fitted'],'r-+')
                pl.draw()

        #the result is g*alpha for the simple case
            omlist.append(omega[i])
            glist.append(g)
        #end for i
        omega = asarray(omlist)
        G = asarray(glist)
        dG = asarray(dG)[r0:-r0-1]
    else:
        #if it is the old way, then just combine the phase and the G:
        G = asarray(glist,dtype='complex')*exp(cip*asarray(alphalist))
        omega = asarray(omega)
    #end if advanced

    #now clean up:

    res = dict()
    if flip:
        res['omega'] = omega[::-1]
        res['G'] = G[::-1]
        res['f'] = res['omega']/(2.0*pi)
        res['dG'] = asarray(dG)[::-1]
    else:
        res['omega'] = omega
        res['G'] = G
        res['f'] = res['omega']/(2.0*pi)
        res['dG'] = asarray(dG)


    return res
#end of  J_to_G_Mason

