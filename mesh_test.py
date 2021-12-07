#
from fenics import *
from mshr import *

def mesh_test1 ( ):

#*****************************************************************************80
#
## mesh_test1 creates meshes in several ways.
#
#  Discussion:
#
#    The word "mesh" is sometimes a keyword.  Therefore, we prefer to use
#    the word "my_mesh" for instances of meshes that we create, to avoid
#    confusing statements such as "mesh = mesh".
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    08 November 2018
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt

  print ( '' )
  print ( 'mesh_test1:' )
  print ( '  Generate meshes with various options.' )
  print ( '' )
#
#  Create a mesh on the Unit interval [0,1]
#
  my_mesh = UnitIntervalMesh ( 10 )
  print ( '  Plotting a UnitIntervalMesh' )
  plot ( my_mesh, title = 'UnitIntervalMesh' )
  filename = 'unitintervalmesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Rectangle with SW and NE corners specified and right/left option.
#
  sw = Point ( 0.0, 0.0 )
  ne = Point ( 10.0, 4.0 )
  my_mesh = RectangleMesh ( sw, ne, 5, 2, 'right/left' )
  print ( '  Plotting a RectangleMesh' )
  plot ( my_mesh, title = 'RectangleMesh')
  filename = 'rectanglemesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Unit cube.
#
  my_mesh = UnitCubeMesh ( 4, 2, 3 )
  file = File("Unit_cube_mesh")
  file << my_mesh
  print ( '  Plotting a UnitCubeMesh' )
  #plot ( my_mesh, title = 'UnitCubeMesh' )
  #filename = 'unitcubemesh.png'
  #plt.savefig ( filename )
  #print ( '  Graphics saved as "%s"' % ( filename ) )
  #plt.close ( )
  #print ( '  UnitCubeMesh created %d cells.' % ( my_mesh.num_cells ( ) ) )
#
#  Use MSHR to mesh a circle minus rectangle.
#
  c = Point ( 0.0, 0.0 )
  r = 5.0
  sw = Point ( -3.0, -3.0 )
  ne = Point ( -2.0, 2.0 )
  domain = Circle ( c, r, segments = 16 ) \
         - Rectangle ( sw, ne ) 
  my_mesh = generate_mesh ( domain, 16 )
  print ( '  Plotting a Circle Minus Rectangle Mesh' )
  plot ( my_mesh, title = 'Circle-Rectangle Mesh' )
  filename = 'circleminusrectanglemesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Read a mesh from an XML file 
#
  my_mesh = Mesh ( 'cape.xml' )
  print ( '  Plot a mesh created from an XML file' )
  plot ( my_mesh, title = 'cape.xml' )
  filename = 'cape.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
  print ( '  cape.xml created %d cells.' % ( my_mesh.num_cells ( ) ) )
#
#  Terminate.
#
  return

def mesh_test2 ( ):

#*****************************************************************************80
#
## mesh_test2 creates a mesh and queries it.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    08 November 2018
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt

  print ( '' )
  print ( 'mesh_test2:' )
  print ( '  Create just one mesh, and query it.' )
  print ( '' )
#
#  Rectangle with SW and NE corners specified and right/left option.
#
  my_mesh = UnitSquareMesh (2, 3, 'right/left' )
  print ( '  Plotting an example mesh.' )
  plot ( my_mesh, title = 'example mesh')
  filename = 'example_mesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Request the number of items.
#
  print ( '' )
  print ( '  my_mesh.num_vertices ( ) = %d.' % ( my_mesh.num_vertices ( ) ) )
  print ( '  my_mesh.num_cells ( ) = %d.' % ( my_mesh.num_cells ( ) ) )
  print ( '  my_mesh.num_edges ( ) = %d.' % ( my_mesh.num_edges ( ) ) )
  print ( '  my_mesh.num_faces ( ) = %d.' % ( my_mesh.num_faces ( ) ) )
  print ( '  my_mesh.num_facets ( ) = %d.' % ( my_mesh.num_facets ( ) ) )
#
#  Repeat call, after init().
#
  print ( '' )
  print ( '  Call my_mesh.init(), which defines the extra information.' )

  my_mesh.init ( )

  print ( '' )
  print ( '  my_mesh.num_vertices ( ) = %d.' % ( my_mesh.num_vertices ( ) ) )
  print ( '  my_mesh.num_cells ( ) = %d.' % ( my_mesh.num_cells ( ) ) )
  print ( '  my_mesh.num_edges ( ) = %d.' % ( my_mesh.num_edges ( ) ) )
  print ( '  my_mesh.num_faces ( ) = %d.' % ( my_mesh.num_faces ( ) ) )
  print ( '  my_mesh.num_facets ( ) = %d.' % ( my_mesh.num_facets ( ) ) )
#
#  Request hmin, hmax, and ufl_cell.
#
  print ( '' )
  print ( '  my_mesh.hmin() = %g' % ( my_mesh.hmin ( ) ) )
  print ( '  my_mesh.hmax() = %g' % ( my_mesh.hmax ( ) ) )
  print ( '  my_mesh.ufl_cell() = %s' % ( my_mesh.ufl_cell ( ) ) )
#
#  Get mesh vertex coordinates.
#
  x = my_mesh.coordinates ( )

  print ( '' )
  print ( '  my_mesh.coordinates():' )
  print ( x )
#
#  Get element-to-vertex connectivity.
#
  etov = my_mesh.cells ( )

  print ( '' )
  print ( '  my_mesh.cells():' )
  print ( etov )
#
#  Terminate.
#
  return

def mesh_test ( ):

#*****************************************************************************80
#
## mesh_test demonstrates some features of meshes.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    08 November 2018
#
#  Author:
#
#    John Burkardt
#
  import time

  print ( time.ctime ( time.time() ) )
#
#  Report level = only warnings or higher.
#
  level = 30
  set_log_level ( level )

  print ( '' )
  print ( 'mesh_test:' )
  print ( '  FENICS/Python version' )
  print ( '  Demonstrate some properties of meshes.' )

  mesh_test1 ( )
  mesh_test2 ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'mesh_test:' )
  print ( '  Normal end of execution.' )
  print ( '' )
  print ( time.ctime ( time.time() ) )
  return

if ( __name__ == '__main__' ):

  mesh_test ( )
