from fenics import*
import fenics as fe
from dolfin import*
import matplotlib.pyplot as plt
# Create mesh and define function space
mesh = fe.UnitCubeMesh(8, 8, 8)
V = fe.FunctionSpace(mesh, "Lagrange", 2)
# Define boundary conditions
u0 = fe.Expression("1 + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]",degree=1)
def u0_boundary(x, on_boundary):
    return on_boundary
bc = fe.DirichletBC(V, u0, u0_boundary)
# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
a = fe.inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx
# Compute solution
u = fe.Function(V)
fe.solve(a == L, u, bc)
# Plot solution and mesh
plt.figure()
fe.plot(u)
plt.figure()
fe.plot(mesh)
# Dump solution to file in VTK format
file = File("poisson3d.pvd")
file << u
# Hold plot
#plt.interactive()
plt.show()
plt.show()
