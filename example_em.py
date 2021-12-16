#Tomado de https://scicomp.stackexchange.com/questions/32844/electromagnetism-fem-fenics-interpolation-leakage-effect

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import mshr
import copy
from scipy import constants
#############################################################################
import time
#############################################################################
from ProcessSubDomains import *
#############################################################################
mesh = Mesh('Mesh.xml')
# ---------------------------------------------------------------------------
markSubdomains('subdomains.xml', subdom_filename) # custom function that marks each subdomain with a number
materials = MeshFunction('size_t', mesh, mesh.topology().dim())
File(subdom_filename+'.xml') >> materials

#############################################################################
# MATERIAL PROPERTIES
class MaterialProperty(UserExpression):
    def __init__(self, materials, property, material_subdomains, property_index, **kwargs):
        super(MaterialProperty, self).__init__(**kwargs)    
        self.materials = materials
        self.property = property
        self.material_subdomains = material_subdomains
        self.property_index = property_index

    def eval_cell(self, values, x, cell):
        label = self.materials[cell.index]
        for key in self.material_subdomains:
            if label in self.material_subdomains[key]:
                values[0] = self.property[ self.property_index[key] ]

print("Assigning material properties...")
material_subdomains = {'air': [0, 2], 'iron': [1], 'wire': [3, 4, 5, 6]} # numbers associeted with subdomains
property_index = {'air': 0, 'iron': 1, 'wire': 2} # property index in permeability array
permeability = constants.mu_0*np.array([1, 35e3, 0.9991])
mu = MaterialProperty(materials, permeability, material_subdomains, property_index, degree=0)

#############################################################################
# DEFINE FUNCTION SPACE
V = FunctionSpace(mesh, 'P', 1)

#############################################################################
# BOUNDRY CONDITIONS
tol = 1e-14
def boundry(x, on_boundry):
    return on_boundry and ( near(abs(x[0]), 0.1, tol) or near(abs(x[1]), 0.1, tol) ) # checking when on Dirichlet BC
u_D = Constant(0) # value on Dirichlet BC
bc = DirichletBC(V, u_D, boundry)

#############################################################################
# REDEFINE INTEGRATION MEASURES
dx = Measure('dx', domain = mesh, subdomain_data = materials)

#############################################################################
# SOURCES
I = 400000.0
J_A = Constant(I)
J_B = Constant(I)

#############################################################################
# TRIAL AND TEST FUNCTIONS
# TRIAL AND TEST FUNCTIONS
A_z = TrialFunction(V)
v = TestFunction(V)

#############################################################################
# SOLVE VARIATIONAL PROBLEM
print("Solving variational form...")
a = (1/mu)*inner( grad(A_z), grad(v) )*dx
L = J_A*v*dx(3) - J_A*v*dx(4) + J_B*v*dx(5) - J_B*v*dx(6)

A_z = Function(V)
solve( a == L, A_z, bc)

# POSTPROCESSING
# calculate derivatives
Bx = A_z.dx(1)
By = -A_z.dx(0)

B_abs = np.power( Bx**2 + By**2, 0.5 ) # compute length of vector

# define new function space as Discontinuous Galerkin
abs_B = FunctionSpace(mesh, 'DG', 0)
f = B_abs # obtained solution is "source" for solving another PDE

# make new weak formulation
w_h = TrialFunction(abs_B)
v = TestFunction(abs_B)

a = w_h*v*dx
L = f*v*dx

w_h = Function(abs_B)
solve(a == L, w_h)

# plot the solution
plot(w_h)
plt.show()
