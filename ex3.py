from fenics import *
import fenics as fe
from dolfin import*
from ufl import nabla_grad
from ufl import nabla_div
import matplotlib as plt
from mpl_toolkits import mplot3d

# Scaled variables
L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Define strain and stress

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_ * nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)
print(u)

# Dump solution to file in VTK format

#Plot solution
#fe.plot(u, title='Displacement', mode='displacement')

#Plot stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
#fe.plot(von_Mises, title='Stress intensity')


# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
file = File("stress2.pvd")
file << u_magnitude
#fe.plot(u_magnitude, 'Displacement magnitude')
#print('min/max u:',
#      u_magnitude.vector().array().min(),
#      u_magnitude.vector().array().max())
file = File("stress.pvd")
file << u

file = File("stress1.pvd")
file << von_Mises

file = File("stress2.pvd")
file << u_magnitude
