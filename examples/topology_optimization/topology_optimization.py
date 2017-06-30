from matplotlib import pyplot
from scipy import integrate
import numpy

# VARIABLE DECLARATIONS

# Create the 2d space for our problem
nx = 41
ny = 41
nt = 500
dx = 2. / (nx - 1) 
dy = 2. / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

# These are the heat conduction coefficients of material 1 and 2 respectively.
heat_conduction1 = 0 
heat_conduction2 = 0 

vol_heat_source = 0
temperature = 0


# This is the Lagrange multiplier, lambda
lagrange_mult = 0

# we'll need to update this during the solving processlater
# heat_flux1 = heat_conduction1 * temperature.gradient
# heat_flux2 = heat_conduction2 * temperature.gradient

#
