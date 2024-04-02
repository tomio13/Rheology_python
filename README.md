This is a small python package to do rheology related calculations in passive
particle tracking microrheology.

Requirements:
- python 2.7
- numpy 1.5
- scipy 0.10
- matplotlib    1.0

It may also work with some earlier versions, but these have been used for
testing.

Capabilities:
 - calculating mean squared displacement from positions
 - higher order displacements
 - maximum mean excursion (quadratic and higher as well)
 - creep-compliance from MSD
 - complex shear modulus from creep compliance using various methods

 The last one can be done using the method proposed by Mason et al. and
 Dasgupta et al., and the numerical transform proposed by Evan et al. There is
 an improved method as well, using local fitting and then a dynamically
 refined data set to get the complex shear modulus.

Mason's method is limited to power functions, so be careful using it. The
Evans method is very sensitive to data noise and to sampling error at high
frequenxies (producing a cut off).

An example script  and general tool is provided in the Examples under the
ProcessRheology.py. It uses a config file to receive commands, there is a
commented example in the same folder.

The full description of the configuration is in the config.txt.example file,
and an example dataset and correscponding config file is also provided in the
Examples folder (fov2191wDD.txt, timestamp-table.txt, config.txt). The example
data is an actin network on micropillars bundled using a testbuffer containing
8mM magnesium (from 25th July 2012, Timo Maier).

The Function-test.py runs some theoretical tests and shows the results
graphically. It is a useful tool to test the functions and see how the various
results change when one varies the parameters (see the comments within).


Install:
python setup.py build
and
sudo python setup.py install

should do the trick.

Usage: the common import utility will work (from Rheology import *).

I hope you find these tools useful and interesting. Any further improvements
and suggestions are welcome.

2012 June, Heidelberg       T.H.
