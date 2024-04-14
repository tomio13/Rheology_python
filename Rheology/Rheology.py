#!/usr/bin/env python
""" A set of routines to calculate parameters for particle microrheology or
    diffusion analysis of particles.
    The methods include:
        mean square displacement from time average
        mean maximum excursion
        creep-compliance and complex shear modulus

    Author:     Tomio
    Date:       2011 Jan.
    Warranty:   None
    License:    LGPLv3
"""

import numpy as nu
from numpy import linalg
from matplotlib import pyplot as pl
from PowerFit import MSD_power_fit, MSD_interpolate, f_pow, f_Kelvin,\
                    fit_pow, fit_Kelvin, J_to_G_Mason
from RotateDataset import RotateDataset

###########################################################
__all__= ["GetData", "TimeAverageShift", \
        "MSD", "MKD", "MSD_to_J", "J_to_G",\
        "J_to_G_Mason",\
        "PBlowingSphere", "EstimateR0",\
        #from PowerFit:
        "f_pow", "f_Kelvin", \
        "fit_pow", "fit_Kelvin",\
        "MSD_power_fit", "MSD_interpolate",\
        #and more:
        "RotateDataset" ]
__version__="1.0"

##########################################################

#add a key list defining: [indx, x, y, z,...] or allow a list to be used...
def GetData(ts, poslist, indx=0, order=1, resolution=1.0, \
        tscale= 1.0, keylist=["indx","X","Y"], \
        Nd = 1000, verbose=False):
    """ get the position data of one particle from the time stamps and
        position list. The position list is a list of dicts, which may
        hold 1,2,3 (N) dimensional trajectory data. Try being flexible,
        allowing the usage of various keys.

        The structure of poslist: it is a list of trajectory data for
        each particles [particle0, particle1, ...]. Each particle dataset
        is a dict, holding tracking informations, such as:
        particle0 = {'X','Y','Z','imageindex','sum indensity',...}

        Because multiparticle tracking may result in merged lists of
        tracks, the algorithm expects an index list. Only those positions
        are taken where this index lists has the value of indx in the
        input parameters.

        Drift correction: this is implemented by first smoothening
        the data with an Nd range running average, then fittin an
        'order' order polynomial to the smoothened data. If 'order'
        is -1, do nothing.
        This way the polynomial may ensure that the bandwidth is
        minimally affected.

        Parameters:
        ts:         timestamp array or None
        poslist:    a position data set as saved from tracking
        indx:       which particle to return, an index
        order:      drift correction with a polynomial of this order
                    if -1 then subtract the first position only
                    and ignore Nd.
                    (Linear drift correction is commonly applied.)

        resolution: microns/pixel (the array is multiplied by this value)
        tscale:     seconds/image. Scaling index if ts is None
                    (if ts is provided, it is not used!)

        keylist:    which keys are decipting the image index and the
                    various coordinates?
                    By default we look for "indx", "X", "Y", and "Z".

        Nd:         apply a running average before the polynomial
                    fit, with Nd width. If Nd < 2: use N/500
                    if Nd== -1, turn this off

        verbose:    print(some information)
        """

    if len(keylist) < 2:
        print("Need at least the index and a position key!")
        return ([],[])

    if indx < 0 or indx  > len(poslist):
        print(f'invalid index: {indx}')
    #end if

    #default time array is based on the indexes:
    indxkey = keylist[0]
    if poslist[indx].has_key(indxkey):
        i = nu.asarray(poslist[indx][indxkey])
    else:
        print("invalid index key!")
        raise ValueError

    #we need an x,y plot which is time vs. position:
    x = i.astype(nu.float) * tscale if tscale > 0.0 else i.astype(nu.float)

    #default x is set, overwrite it if ts exists:
    if ts != None:
        try:
            x = nu.asarray(ts)
        except:
            if verbose:
                print("unable to convert time stamps to time array")
        else:
            if verbose:
                print("timestamps are collected")

    #shift to the first index for comfort:
    x = x - x[0]

    #keylist[1:] holds the position keys. If this fails, anyway
    #we have a problem, so let python raise the exception:
    #check the keys:
    kl = keylist[1:]
    for j in kl:
        if not poslist[indx].has_key(j):
            keylist.remove(j)
    #end for
    if len(keylist) < 2:
        print("No positions left!")
        raise ValueError

    N = len(poslist[indx][keylist[1]])
    N2 = len(keylist)-1
    #our return array will be N x spatial dimensions in size:
    a = nu.zeros((N,N2))

    #now fill it up:
    for j in range(N2):
        a[:,j] = nu.asarray(poslist[indx][keylist[j+1]])

    if order == -1:
        const = a[0,:]
        a = a - const

        if verbose:
            print("Subtracted:", const)

    elif order >= 0:
        if verbose:
            print("subtracting drift using %d order polynomial" %order)

        if Nd != -1 and (Nd < 2 or Nd > N):
            Nd = N/500
            if verbose:
                print("Invalid Nd! New value is set to: %d" %Nd)
            #end if

        #Running average:
        b = a.copy()

        #do the running averaging.
        #mean(axis=0) ensures that all coordinate columns are
        #processed in one step.
        if Nd > 0:
            for i in range(N-Nd):
                b[i,:] = a[i:i+Nd].mean(axis=0)
            for i in range(1,Nd):
                b[-i,:] = a[-i-Nd:-i, :].mean(axis=0)

        #now do a fitting to b, but correct a!
        #nu.polyfit can deal with 2D
        #shifting x may improve the accuracy of polyfit (see its help)
        xfit = x - x.mean(axis=0)
        for j in range(a.shape[1]):
            fit = nu.polyfit(xfit, b[:,j], order)
            a[:,j] = a[:,j] - nu.polyval(fit, xfit)
            if verbose:
                for k in range(len(fit)):
                    print("%d: Polygon coefficient %d: %.5f" %(j,k, fit[k]))

    else:
        if verbose:
            print("invalid order")

    if resolution != 0.0 and resolution != 1.0:
        a = a* resolution
        if verbose:
            print("Scaled data with: %f" %resolution)

    return (x, a)
#end GetData

def TimeAverageShift(pos, tau, tvector=None, tolerance=0.1,\
                        overlap=False, sum=True, Corr=False,#
                        MME=False):
    """ Calculate the steps between datapoint pairs within a trajectory.
        The function calculates the step if the index of the positions is
        shifted with 'tau'. This data can be used to calculate time averaged
        information, such as MSD, MKD or MME.
        Depending on the settings, the function returns the coordinates of
        the step or the square of the distances. The calculation is either
        performed on non-overlapping or overlapping intervals (see parameters
        below).

        Parameters:
        pos:        1D or 2D data set of positions. If 2D, then the second
                    index is treated as coordinates.
        tau:        a single number for delay. It is an index, no time, thus
                    an integer >1 is expected.
        tvector:    If provided, it should be the time point of each data line
                    in pos. Based on its values the time shift for each pair of
                    points is calculated and tested.
                    First a mean is estimated then those deviating too much are
                    rejected.
        tolerance:  0 - 1.0 used to calculate the relative tolerance in the
                    time shifts. (1-tolerance)*mean ... (1+tolerance)*mean will
                    be accepted.
                    If you do not want check, set tvector to None

        sum:        if True, sum up the square of the coordinates
                    if False, return the coordinates of the steps unchanged

        Corr:       if True,  the data is meant for correlating trajectories
                    thus we need the (dx,dy...) set and dt if available
                    sum is overriden to False.

        MME:        find the most distant point in t:t+tau interval,
                    and get its coordinates dx,dy,...
                    Return value depends on 'sum' above.
                    This is slow, since it has to check for each interval
                    individually. Especially slow with overlap=True.
                    (overrides Corr)

        return:
        a dict of arrays, depending on the input parameters
        "RSQ":  squared distance (sum= 1, Corr= 0)
        "R" :   if sum= 0
        "dt":   array of delay values (tvector!= None)
        "tau":  the delay (dt.mean() or the index step)
        "dtau": dt.std()
        "t0":   the start time point of the given data (Corr= 1)
        "DR":   dr array (Corr= 1) [dx,dy,...]
    """
    if pos.ndim == 1:
        pos.shape = (pos.shape[0],1)

    elif pos.ndim > 2:
        print("Invalid data set!")
        return None

    N = pos.shape[0]
    # make sure:
    tau = int(tau)

    #let us deal with the tvector:
    if tvector != None:
        if type(tvector).__name__ != 'ndarray':
            print("Invalid time data, a numpy array is expected!")
            return None
        else:
            if tolerance < 0.0 or tolerance > 1.0:
                print("invalid tolerance. It should be in [0,1].")
                return None
            #end if
        #end if
    #end if tvector

    if tvector != None and len(tvector) != N:
        print("Invalid length of tvector!")
        return None
    #end if

    if tau > N-1 or tau < 1:
        print("Invalid delay value! Possible limits are: 1 - %d" % (N-1))
        return None
    #end if

    #now the results:
    res = dict()

    #Easy life is over, we have to deal with an array of time points
    #and define our dt in them with a tolerance
    if overlap:
        M = N-tau
        steps = 1
        i0 = 0
        j0 = tau
        i1 = M
        j1 = N

    else:
        M = int(N/tau)
        steps = tau
        i0 = 0
        j0 = tau
        i1 = (M-1)*tau
        j1 = min(M*tau,N)

    #print("Number of data:",M)
    #print("indices: %d:%d %d:%d step: %d" %(i0,i1,j0,j1,steps))
    #i0:i1:steps define the intervals (overlapping or not)
    #each step is between i and i+tau, and i runs in i0:i1:steps
    if MME:
        indx = range(i0,i1,steps)
        #this r is an array of coordinates:
        r = nu.zeros((len(indx), pos.shape[1]))
        j=0
        #What is the maximal distance in each i:i+tau segment?
        for i in indx:
            #we walk through each segments and get the maximum
            #distance migrated within the segment:
            dr = pos[i:(i+tau+1)] - pos[i]
            #which is the most distant?
            dr2 = (dr*dr).sum(axis=1)
            #take the maximal distance (if there are more, chose the
            #first one):
            ii = ((dr2 == nu.max(dr2))).nonzero()[0]
            #then we provide the coordinates of the given step:
            r[j] = dr[ii,:]
            j += 1
        #end for i
    else :
        r = pos[j0:j1:steps,:] - pos[i0:i1:steps,:]

    if tvector != None:
        #now do the screening:
        # - we need the time shift:
        dt = tvector[j0:j1:steps] - tvector[i0:i1:steps]
        tm = dt.mean()
        tup = (1.0+tolerance)*tm
        tlow = (1.0-tolerance)*tm

        #this is a set of indices we can use:
        #print(tlow,":",tup)
        indx = ((dt >= tlow) & (dt <= tup))
        #print("Dropped %d points" %(indx.size-indx.sum()))
        r = r[indx,:]
        dt = dt[indx]
        res['dt'] = dt
        res['tau'] = dt.mean()
        res['dtau'] = dt.std()
    else:
        res['tau'] = float(tau)

    #if the user request a distance and not the square of each
    #coordinate, we do sum up along the coordinates
    #of course, when there is only 1 coordinate,
    #then there is nothing to do
    if not MME and Corr:
        res['DR'] = r

    r = r*r

    if sum and pos.shape[1] > 1:
        res['RSQ'] = r.sum(axis=1)

    elif sum:
        #for 1D problems:
        res['RSQ'] = r

    else:
        res['R'] = r

    return res
#end of TimeAverageShift

def MSD( pos, tau=0, tvector=None, tolerance=0.1,\
        overlap=False, MME=False):
    """ Calculate the time averaged mean square displacement from position
        data. The routine assumes an equidistant sample of data as rows of
        the input data set, but applies corrections if tvector is provided.
        The columns are the coordinates (x,y,z).

        The algorithm uses the time average of overlapping data, thus
        1/(N-j) * sum (from 0 to (N - j): ||pos[i+j] - pos[i]||)

        Missing time points are handled by TimeAverageShift if the vector
        of time points (in floating point, possibly seconds) is provided.

        Parameters:
            pos:        position data, 1 or 2D array (2nd index is coordinates)
            tau:        delay settings
                        if 0, then generate delay up to N/4
                        if a number > 0 then an array is generated up to
                        that number (1...tau)
                        if a list or 1D array, then use the values there
                        the minimum should be > 1, the maximum < N/4

            tvector:    a vector of time stamps or None
                        if provided, the delay will be tested for each data
                        point. See: TimeAverageShift() for more details.

            tolerance:  relative tolerance in time shift. This much deviation
                        from the average is tolerated.

            overlap:    if TRUE, the data ranges may overlap, if False, then
                        non-overlapping ranges are created (if tau is a number
                        only). If tau is specified, this parameter is not
                        used.

            MME:        if True, use the mean of the maximum distance between
                        eacht, t+tau intervals (mean maximum excursion)

        Return:
            a dict containing the results
            MSD:  the mean squared displacement or mean maximum excursion
                    (dr**2)

            tau:  the average delay values
                    (equals to the input parameter tau, if tvector is None)

            if tvector is defined (not None):
            dtau: the standard deviation of the delays

            DMSD: the standard error estimated to MSD for each delays.
                  (this value is just a guiding value, statistically it
                  is not correct to calculate from overlapping intervals!)
                    This is not provided if MME is set
    """
    N = pos.shape[0]

    #a dirty trick
    if pos.ndim == 1:
        pos.shape = (pos.shape[0], 1)

    elif pos.ndim > 2:
        print("The input should be a 1 or 2 dimensional array!")
        return None

    #tau is the size of the index jump
    #after it is used we calculate what it means in time
    #only if tvector is provided
    ttype = type(tau).__name__
    if ttype == "list" or ttype=="ndarray":
        taus = list(tau)
    else:
        tau = tau if (tau > 0 and tau < N) else int(N/4)
        taus = range(1,tau)

    Nt = len(taus)
    res = dict()
    res['MSD'] = []
    res['DMSD'] = []
    res['N'] = []
    res['tau'] = []

    res['dtau'] = []

    #sometimes TimeAverageShift returns empty data sets
    #using lists will prevent the contamination of the results

    #First scan through the data:
    for j in range(len(taus)):
        t = taus[j]

        #get the distance data with delay t:
        r = TimeAverageShift(pos, t,\
             tvector= tvector, tolerance= tolerance,\
             overlap= overlap, sum= True, Corr= False, MME= MME)

        if r['RSQ'].size < 1:
            print("Empty array at delay index: %.1f" %t)

        else:
            rs= r['RSQ'].mean()
            res['MSD'].append(rs)
            N = r['RSQ'].size
            res['N'].append(N)

            #recycle rs for the std. error:
            rs= 0 if MME else r['RSQ'].std()/float(N)
            res['DMSD'].append(rs)

            res['tau'].append(r['tau'])

            dtau = r['dtau'] if tvector != None else 0.0
            res['dtau'].append( dtau )
    #end for

    #for the SSE compatibility in numpy 1.8, we need this asarray:
    for i in res.keys():
        res[i] = nu.ascontiguousarray( res[i])

    #print("Max tau:", res['tau'].max())
    return res
#end of MSD



def MKD(pos, k= 2, tau= 0, tvector= None, tolerance= 0.1, \
        overlap= False, MME= False):
    """ Calculate the mean displacement of the k-th power from position data.
        The routine assumes an equidistant sample of data as rows of
        the input data set. (see MSD)

        The algorithm uses the time average of overlapping data, thus
        1/(N-j) * sum (from 0 to (N - j): ||pos[i+j] - pos[i]||)

        To avoid problems with negative numbers, the algorithm takes
        the square of the step, then raises to the k/2-th power.

        Parameters:
            pos:        position data, 1, 2 or 3D data set
            k:          the power to use: it should be a positive integer
                        or anything

            tau:        delay settings
                        if 0, then generate delay up to N/4
                        if a number > 0 then an array is generated up to
                        that number (1...tau)
                        if a list or 1D array, then use the values there
                        the minimum should be > 1, the maximum < N

            tvector:    a vector of time stamps or None
                        if provided, the delay will be tested for each data
                        point. See: TimeAverageShift() for more details.

            tolerance:  relative tolerance in time shift. This much deviation
                        from the average is tolerated.

            overlap:    if TRUE, the data ranges may overlap, if False, then
                        non-overlapping ranges are created (if tau is a number
                        only). If tau is specified, this parameter is not
                        used.

            MME:        if True, use the maximum excursions for the mean

        Return:
            a dict containing the results
            "MKD"       the mean k-th momentum of displacement
            "tau"       the delay points for the MKD
                        (equals to the input parameter tau, if tvector is None)

            "dtau"      the standard deviation of the delays
                        only returned if tvector is provided (not None)

            "DMKD":     the standard error estimated to MSD.
                        (set to 0 if MME is True)

            (MKD,stdev,N) the values, a standard deviation and number of data
                          points used for each value. If MME is True, stdev
                          is 0.
    """

    N = pos.shape[0]
    pos = pos.astype(nu.float)

    #a dirty trick
    if pos.ndim == 1:
        pos.shape = (pos.shape[0],1)

    elif pos.ndim > 2:
        print("The input should be 1 or 2 dimensional array!")
        return None

    ttype = type(tau).__name__
    if ttype == "list" or ttype=="ndarray":
        taus = list(tau)
    else:
        tau = tau if (tau > 0 and tau < N) else int(N/4)
        taus = range(1, tau)

    if k == 0:
        print("k=0!! Use numpy.ones instead! k should be nonzero")
        return nu.ones(len(taus))

    k = 0.5*float(k)
    res = dict()
    Nt = len(taus)
    res['MKD'] = []
    res['DMKD'] = []
    res['N'] = []
    res['tau'] = []
    res['dtau'] = []

    #First scan through the data:
    for j in range(Nt):
        t = taus[j]
        r = TimeAverageShift(pos, t, \
                tvector= tvector, tolerance= tolerance,\
                overlap= overlap, sum= False, Corr= False, MME= MME)

        #we want the position average, but not averaging for the
        #dimensions. However, the length may be shorter or longer,
        #TimeAverageShift is returning ['R'], which contains
        #the square of coordinates for each point (delta r).
        #
        #so, we do the k-th power and sum up along the coordinate dimension:
        if r['R'].size < 1:
            print("Empty array at delay index %.1f" %t)

        else:
            #each coordinate has to he put to the kth power first
            #then sum them up:
            rk = (r['R']**k).sum(axis=1) if k != 1 else r['R'].sum(axis= 1)

            res['N'].append(rk.size)

            rs = rk.mean()
            res['MKD'].append(rs)
            rs = 0.0 if MME else rk.std()/float(rk.size)
            res['DMKD'].append(rs)

            #generate return values:
            res['tau'].append( r['tau'] )

            dtau = r['dtau'] if tvector != None else 0.0
            res['dtau'].append(dtau)
    #end for

    for i in res.keys():
        res[i] = nu.asarray(res[i], dtype=nu.float64)

    return res
#end of MKD

def PBlowingSphere(pos, tau, r0= 1.0, alpha= 1.0, \
                    tvector= None, tolerance= 0.1,\
                    overlap= False, histogram= False):
    """ Calculate the probability of a particle to be at distances:
        r0*t^alpha, for a set of t values, specified by tau.
        Uses TimeAverageShift to generate the distances for a delay of t,
        and then calculates N( r**2 < r0**2*t**alpha) / N.
        (the original is r < r0*t**(alpha/2))

        From: Tejedor et al. Biophysical Journal vol.98:1364 - 1372 (2010)

        Parameters:d
        pos:        array of positions (time series), 1D or 2D array. If
                    2D is provided, then the second index is treated as
                    coordinates.
        tau:        if 0 then  use t= 1...N/4, if 0< t <= N then use 1...t,
                    if a list or array, then use it as is.
                    (delay defined in index steps. If tvector is not None,
                    this is converted to real time delay.)

        r0:         radius to be tested. It should be identified from the
                    position data first by the user.

        alpha:      slope of the MSD in log-log plot, user defined

        tvector:    the vector of time stamps to be used by TimeAverageShit
                    (optional)

        tolerance:  see TimeAverageShift

        overlap:    by default non-overlapping sampling should be used
                    overlapping may mess up the statistics.
        histogram:  bool; provide the histogram? min(N/10, 50) bins are used.

        Return:
            a dictionary of the results
            "P":    the probability values
            "N":    the number of data points used to get these P values
            'tau':  the final delay values used (if tvector is specified,
                    this definitely differs from the input tau)

            if histogram is True:
                Both of these are 2D, one set of bins or histogram values per
                delay value
            "bins"  the bins of the histogram
            "hist"  a cumulative histogram values normalized to maximum= 1.
    """

    N = pos.shape[0]
    pos = pos.astype(nu.float)

    #a dirty trick
    if pos.ndim == 1:
        pos.shape = (pos.shape[0],1)

    elif pos.ndim > 2:
        print("The input should be 1 or 2 dimensional array!")
        return None

    ttype = type(tau).__name__
    if ttype == "list" or ttype== "ndarray":
        taus = list(tau)
    else:
        tau = tau if (tau > 0 and tau < N) else int(N/4)
        taus = range(1, tau)

    #we want to compare the squared position, using the MSD:
    r0 = r0 * r0
    res = dict()
    Nt = len(taus)

    res['P'] = nu.zeros(Nt, dtype= nu.float)
    res['N'] = nu.zeros(Nt, dtype= nu.float)
    res['tau'] = nu.zeros(Nt, dtype= nu.float)

    if histogram:
        res['bins'] = []
        res['hist'] = []

    #First scan through the data:
    for j in range(len(taus)):
        t = taus[j]

        r = TimeAverageShift(pos, t,\
                tvector= tvector, tolerance= tolerance,\
                overlap= overlap, sum= True, Corr= False)

        #we requested a delay, but the real one is r['tau']
        #if tvector is not None:
        if r['tau'].size < 1:
            print("Empty array at index: %.1f" %t)
            return None

        else:
            t = r['tau']
            limit = r0*(t**alpha)
            indx = (r['MSD'] <= limit)

            res['tau'][j] =  float(t)
            res['N'][j] = len(r['MSD'])
            res['P'][j] = float(indx.sum()) / float(Ns)

            if histogram:
                hist, bins = nu.histogram(r, min(len(r)/10, 50), normed=False)
                bins = 0.5*(bins[1:] + bins[:-1])
                res['hist'].append(hist.cumsum().astype(nu.float)/hist.sum())
                res['bins'].append(bins)
        #end if r['tau']
    #end of for

    return res
#end of PBlowingSphere

def EstimateR0(pos, p= 0.5, \
                tvector= None, tolerance= 0.1,\
                overlap= False):
    """ Put up a cumulative histogram of positions based on the first
        MSD data (shift of the points with delay of 1 frame).

        Based on the numpy.histogram command, and the TimeAverageShift() above.

        A good starting R0 is something like the 50% distance.

        Parameters:
        pos:    list of positions, 1D or 2D data
        p:      probability at which estimate R0.
                A linear interpolation is used.

        tvector and tolerance are passed to TimeAverageShift

        overlap:    should the calculation use overlapping data frames?

        Return:
        R0, bins, histogram:
                R0:         a limit value, which encloses part p of the
                            steps in after 1 frame (the displacement is
                            less than R0 for part p of the positions).

                bins:       the middle values of the bins
                histogram:  the normalized cumulative histogram
    """

    N = pos.shape[0]
    pos = pos.astype(nu.float)

    #a dirty trick
    if pos.ndim == 1:
        pos.shape = (pos.shape[0],1)

    elif pos.ndim > 2:
        print("The input should be 1 or 2 dimensional array!")
        return None

    rm = TimeAverageShift(pos, 1, \
            tvector = tvector, tolerance= tolerance,\
            overlap=overlap, sum=True)

    if rm['RSQ'].size < 1:
        print("Error! empty array for delay 1")
        return None

    r = nu.sqrt(rm['RSQ'])

    h, bins= nu.histogram(r, min(r.size/10, 50), normed=False)
    bins = 0.5*(bins[:-1]+bins[1:])
    hs = h.cumsum().astype(nu.float) / h.sum()

    #estimate R0, but we are indexing from 0
    #then the sum of hits is 1 more than the real index
    i0 = max( (hs < p).sum() - 1, 0)
    i1 = min( i0 + 1, bins.size)

    if hs[i0] == hs[i1]:
        print("Warning!!!: Constant histogram at p= %.3f!!!" %p)
        R0 = 0.5*(bins[i0] + bins[i1])

    else:
        #brute force linear interpolation:
        R0 = (p - hs[i0])*(bins[i1] - bins[i0])/(hs[i1] - hs[i0]) + bins[i0]

    return { "R0":R0, "bins":bins, "hist":hs}
#end of EstimateR0

def MSD_to_J(ms, t0=1, tend=299, T=22.0, a=1.0, D=2.0, verbose=False):
    """ Convert the MSD to creep compliance and estimate the two limit:
        at tau=0 and at infinite time delays, thus J0 and eta (viscosity).
       Conversion is done by a constant:
                3*pi*a/(D*kB*T)*scaler (converting micron^2 to m^2)

        MSD is assumed to be in micron**2
        Convert J to SI, thus unit is 1/Pa, and eta in Pas.

        Parameters:
        ms:     an MSD dict, returned by the MSD()
                it is assumed to be in microns**2 units

        t0:     up to what time use the data to t=0 extrapolation (in the
                units of ms['tau'].
        tend:   from what time use the data to calculate eta

        T:      tempearture in Celsius degrees
        a:      particle radius in micron
        D:      dimensionality of the motion (original track)
                (typically 1, 2 or 3)

        Return:
            a dict containing:
            "J":        J values
            "tau":      at which J is calculated
            "eta":      viscosity in Pas
            "J0":       extrapolated J(0) value
            "const":    the conversion multiplier
                        (3 pi a/ (D kB T)) from micron^2 to 1/Pa
            dJ:         if there is a DMSD, const*DMSD
            'a0','b0','a1', 'b1' are the fit parameters

        Based on:
        R. M. L. Evans et al. PRE 80:012501 (2009)
        and T.M. Squires et al. Annu. Rev. Fluid. Mech. 42:413-438 (2010),
            equation 25.
    """

    if not 'MSD' in ms or not 'tau' in ms:
        print("Invalid MSD data")
        return None
    #end if
    res = {}

    if D <= 0.0:
        print("Invalid dimension parameter, falling back to D=2")
        D = 2.0

    #the Boltzmann constant * 1E18 (converting micron**3 to m**3)
    #the 1E5 goes to the numerator in Const to decrease roundoff errors
    #1.3806504E-23 +/- 0.0000024E-23 J/K
    kB = 1.3806504
    #convert T to Kelvin:
    T = T + 273.15

    Const = 3.0 * nu.pi * a * 100000.0/ (D*kB*T)
    if verbose:
        print("Multiplier: %f" %Const)

    #fill up the time part:
    res['tau'] = ms['tau']
    if 'dtau' in ms:
        res['dtau'] = ms['dtau']

    #J and dJ:
    J = Const * ms['MSD']
    res['J'] = J
    res['const']= Const

    #the error scales also linearly:
    if 'DMSD' in ms:
        dJ = Const* ms['DMSD']
        res['dJ'] = dJ

    #Now, we need to extrapolate to t=0 and t= infinity
    indx = ms['tau'] <= t0

    if indx.sum() < 2:
        print("WARNING! Invalid timeframe t0: %f" %t0)

    x = ms['tau'][indx]
    y = J[indx]

    fit = nu.polyfit(x,y,1)
    J0 = fit[1]
    fit0 = fit

    if verbose:
        print("Fit to the beginning %d points: a= %f, b=%f" %(indx.sum(), \
                                                    fit[0], fit[1]))
        print("J0: %f" %J0)

    res['a0'] = fit0[0]; res['b0'] = fit0[1]

    if J0 < 0.0:
        print("Warning: Invalid J0!")
    #end J0
    res['J0'] = J0

    indx = ms['tau'] >= tend

    if indx.sum() < 2:
        print("WARNING! Invalid time frame: %f" %tend)

    x = ms['tau'][indx]
    y = J[indx]
    fit = nu.polyfit(x,y,1)
    eta = 1/fit[0]
    fit1 = fit
    res['a1'] = fit1[0]; res['b1'] = fit1[1]; res['eta']=eta

    if verbose:
        print("Fit to the ending %d points: a= %f, b=%f" %(indx.sum(), \
                                                    fit[0], fit[1]))
        print("eta: %f" %eta)

    return res
#end MSD_to_J

def J_to_G(J, omax=0.0, filter=False, eps=1E-12, Nmax=1000):
    """ Convert J(tau) to G(omega) based on a simple, direct interpolation.

        This function uses an interpolation formula based on the data being
        a set of:
            (A(i) - A(i-1))(t-t(i-1))H(t-t(i-1)),
            where A(i) is the slope of the straight line interpolation,
            H(t) is the Heaviside step function.
            (See the refernces below.)
        The result contains a discrete Fourier transform, thus may be slow,
        but it does not rely on equidistant sampling.

        Parameters:
        J:      a dict generated by MSD_to_J
                J has to contain: J['J'], J['tau'],J['J0'] and J['eta']
                if  J['eta']== 0.0, then J['a1'] or used or that component
                is set to 0.

        omax:   a maximal omega. To limit the bandwidth for oversampled,
                interpolated data. Useful to decrease aliasing effects.
                If <= 0.0 or > 2 pi /dt.min(), then do nothing.

        filter: if True, kill dA[i]/A[i] < eps values

        eps:    the threshold for filter

        Nmax:   maximal number of G data points (sometimes df is just too fine,
                especially for interpolated J data)

        Return:
            a dict containing:

            "G":        a complex array of respons parameters
                        (real part is the storage modulus,
                         imaginary part is the loss modulus)

            "omega":    angular frequency values where this G is interpreted
                        (In estimating the points of f we assume equidistant
                        sampling of tau.)

        Based on the paper
        T. Maier, H. Boehm, T. Haraszti, PRE 86:011501 (2012)
        original method in:
        R. M. L. Evans et al. PRE 80:012501 (2009)
    """
    res={}

    if not 'J' in J or not 'tau' in J:
        print("Invalid input dict.")
        return res

    exp = nu.exp
    complex = nu.complex
    N = len(J['tau'])

    #for the sum formula, we need some work:
    #if tau[0] > 0 (as it should)
    #we insert a tau = 0, where J = J0
    if J['tau'][0] > 0.0:
        #newt = nu.zeros(N+1, dtype=nu.float)
        #newt[1:] = J['tau']
        newt = nu.concatenate((nu.asarray([0.0]), J['tau']))

    else:
        print("Invalid starting point of tau! (<= 0.0)")
        return res

    dt = newt[1:] - newt[:-1]
    if dt.any() <= 0.0:
        print("Invalid t order!")
        return res

    #Extend J as well:
    Js = nu.concatenate((nu.asarray([J['J0']]), J['J']))
    #Js = nu.zeros(N+1, dtype=nu.float)
    #Js[0] = J['J0']
    #Js[1:] = J['J']

    #Frequency range?
    T = nu.max(J['tau'])
    if T <= 0.0:
        print("Invalid time inteval!")
        return dict()

    df = 1.0/T
    #if the time array is irregular, we can have
    #various fmax values:
    Nt = newt.size
    fmax = 0.5*min(1.0/nu.abs(dt.min()), Nt*df)

    #or the user can force one:
    if omax > 0.0:
        fmax = min(fmax, 0.5*omax/nu.pi)
    #end if
    print("T: %.3f s" %T)
    df = max(fmax/float(Nt),df, fmax/float(Nmax))
    print("df: %.3f Hz" %df)
    print("fmax: %.3f Hz" %fmax)

    #frequency:
    #we do not want too many data points. Sometimes df is just too fine,
    #for example, when the J is resampled to very fine points...
    ff = nu.arange(0.0, fmax, df)
    NG = len(ff)
    print("N of G: %d" %NG)
    #allocate the result arrays:
    G = nu.zeros(NG, dtype=nu.complex)
    Jf = nu.zeros(NG, dtype=nu.complex)
    #omega:
    rf = 2.0*nu.pi*ff

    #the slopes:
    #at tau =0 it is 0
    #after the last point it is 1/eta (or a1)
    A = nu.zeros(N+2, dtype=nu.float)
    A[1:-1] = (Js[1:]-Js[:-1])/dt
    A[-1] = J['a1'] if 'a1' in J else J['eta']
    #the difference of the slopes:
    dA = A[1:] - A[:-1]

    #this can go pretty low, and pointless
    #we can decrease the noise cancelling out small ones:
    #A[0] = 0, so we compare to the other side of dA[i]:
    if filter:
        #two steps:
        indx = dA != 0.0
        dA = dA[indx]
        newt = newt[indx]
        newA = (A[1:])[indx]
        #now a divide by zero will not do harm, because it
        #is False in this case:
        indx = nu.abs(dA/newA) > eps
        dA = dA[indx]
        newt = newt[indx]
        print("%d data points after filtering" %len(dA))

    #Go for complex numbers
    dA = dA.astype(nu.complex)
    J0 = complex(J['J0'])

    #iom is i*omega for each omegas
    iom = complex(0.0, 1.0)*rf.astype(nu.complex)
    #the sum of dA[i]exp(-i omega t[i]) is in the denomiator
    #this has to be recalculated for each omega values
    #length is that of len(dA)
    #denom = nu.zeros(N+1, dtype=nu.complex)

    c0 = complex(0.0)
    c1 = complex(-1.0,0.0)

    for i in range(NG):
        iomi = iom[i]
        #-i omega t(i-1)
        #only for ts where dA has a meaning, the rest is 0.0
        fi = c1* iomi*newt
        denom= dA*exp(fi)

        #the part under the denomiator:
        dn = J0*iomi + denom.sum()
        Jf[i] = (J0 + denom.sum()/iomi)/iomi if iomi != c0\
                                                else c1
        G[i] = iomi/dn

    return {"G":G, "omega":rf, "f":ff, 'Jf':Jf, "dA":dA}
#end of J_to_G

