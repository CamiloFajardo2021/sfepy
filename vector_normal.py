from dolfin import *
import numpy as np

def get_facet_normal(bmesh):
    '''Manually calculate FacetNormal function'''

    if not bmesh.type().dim() == 1:
        raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    normals = vec1[:,[1,0]]*np.array([1,-1])
    normals /= np.sqrt((normals**2).sum(axis=1))[:, np.newaxis]

    # Ensure outward pointing normal
    bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]'), degree=1))
    normals[bmesh.cell_orientations() == 1] *= -1

    V = VectorFunctionSpace(bmesh, 'DG', 0)
    norm = Function(V)
    nv = norm.vector()

    for n in (0,1):
        dofmap = V.sub(n).dofmap()
        for i in xrange(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nv[dof_indices[0]] = normals[i, n]

    return norm

mesh = UnitSquareMesh(10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')
n = get_facet_normal(bmesh)

fid = File('normal.pvd')
fid << n
