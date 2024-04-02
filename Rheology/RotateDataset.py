#!/usr/bin/env python
""" If the position distribution is anisotropic, it worths checking an
    orientation parameter. Statistically this can be done looking at the
    distribution as a momentum tensor, and finding its orientation from
    the eigenvectors.
    Then the data can be realigned without losing the generality such,
    that the longer distribution (long axis) is parallel to the Y axis.

    Author:     Tomio
    Date:       2011 Sept.
    Warranty:   None
    License:    LGPLv3
"""

from numpy import *
from numpy.linalg import eig

def RotateDataset(dat, convert=True):
    """ Fit an ellipse using the inertia momentum tensor of the data:
        1. calculate and subtract the center of mass
        2. calculate the inertia momentum tensor
        3. find its eigenvectors
        4. if requested, convert the coordinates by a rotation around
            the center of mass

        Right now it is meant for 2D, but it is possible to extend to 3D
        using two angles of directions and a 3D rotation.

        parameters:
        dat:        2D data set, possibly with 2 columns
        convert:    if True, convert the positions to the new coordinate
                    system.
        return:
        a dict containing
        'Center':           the i,j positions of the center of mass
        'Eigenvector0'
        'Eigenvector1':     direction of the main axes, possibly 0 along
                            the longer one
        'Eigenvals':        a vector of the two eigenvalues
        'MajorAxis':
        'MinorAxis':        the half length of the major and minor axis
        "alpha":            the direction angle of Eigenvector1
                            (hopefully the minor axis)
        'NewData':          the converted data if convert is True
    """
    res = dict()

    if dat.ndim != 2 and dat.shape[1] != 2:
        print("This calculation is designed for 2D data sets having 2 columns")
        return res

    if dat.shape[0] < 3:
        print("Not enough data error!")
        return res

    #Center of mass:
    cent = dat.mean(axis=0)
    res['Center'] = cent

    ndat = dat.copy()
    #recenter the data:
    for i in range(dat.shape[1]):
        ndat[:,i] = dat[:,i] - cent[i]
    #end for i

    #Components of the inertia momentum tensor:
    xx, yy = (ndat*ndat).mean(axis= 0)
    xy = -1.0*(ndat[:,0] * ndat[:,1]).mean()
    common2 = (xx - yy)**2 + 4.0*xy*xy
    common = sqrt(common2)

    #the real x,y turns vs i and j! Thus, x calculated from i is
    #y in real...
    theta = asarray( [[yy,xy],[xy,xx]])

    #Eigen values and vectors:
    thetaeig = eig(theta)
    thetaeigval = asarray(thetaeig[0])
    thetaeigvec = asarray(thetaeig[1])

    if thetaeigval[0] < thetaeigval[1]:
        res['Eigenvals'] = thetaeigval
        res['Eigenvector0'] = thetaeigvec[:,0]
        res['Eigenvector1'] = thetaeigvec[:,1]
    else:
        res['Eigenvals'] = thetaeigval[::-1]
        res['Eigenvector0'] = thetaeigvec[:,1]
        res['Eigenvector1'] = thetaeigvec[:,0]

    #Axis (from Jahne, Digital Image processing, chapter 19, eq: 19.6
    maja = res['MajorAxis'] = sqrt(2.0*(xx+yy+common))
    mina = res['MinorAxis'] = sqrt(2.0*(xx+yy-common))

    #Now, we have a vector and some information about the distribution
    #ellipticity: 1 to infinite, but perhaps 1E12 is a good upper limit
    ell = mina/maja if maja > 1E-8 else 1E12

    if ell > 1.0:
        print("False settings, check the code!")
    elif ell > 0.9 :
        print("Round distribution, nothing to do")
        res['alpha'] = 0.0
    else:
        #rotate the short axis to the X axis:
        e = res['Eigenvector1']
        el = sqrt(e[0]*e[0] + e[1]*e[1])
        e = e/el if el != 1.0 else e
        res['alpha'] = arctan2(e[1],e[0])

        #for a normalized vector, the coordinates are sin and cos:
        cosa, sina = e

        for i in range(ndat.shape[0]):
            x,y = ndat[i,:]
            ndat[i,0] = x*cosa + y*sina
            ndat[i,1] = -x*sina + y*cosa
        #end of for i
        res['NewData'] = ndat
    #end if ellipticity > 1.0

    return res
#end  FitEllipse
