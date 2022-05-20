
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from numba import jit
from numba.typed import List



def compute_raster(first_coord_array, second_coord_array, scalar_values_array,
                   search_radius , operation='nanmedian', raster_meshgrid=None, raster_resolution=1 ):
    '''

    :param first_coord_array:    Nx1 array of first coordinates  (the first_coords and second coords will be usually X,Y but not necessarily
    :param second_coord_array:   Nx1 array of second coordinates
    :param scalar_values_array:  Nx1 array of scalar values  (the scalar_values may be a coordinate (e.g Z) or a property
    :param search_radius:        The radius to search neighbor points of the cloud around the grid_points,
                                 The search is in the first_coord-second_coord plane. The radius is in metric units, not in pixels
    :param operation:            operation to perform on the neighbor points (any of numpy: nanmedian, nanmin, nanmax, nanmean, etc.)
    :param raster_meshgrid:      (optional) 2-tuple of arrays defining the coordinates of the grid for the raster
                                 If not defined, a grid covering all the extension of the cloud at the specified resolution
                                 is created
    :param raster_resolution:    (optional, required if raster_meshgrid is not defined)  In metric units, not in pixels
    :return: raster              array the same size as defined by the raster_meshgrid or the raster_resolution
    '''
    '''
    
    :param first_coord_array:    Nx1 array of first coordinates
    :param second_coord_array:   Nx1 array of second coordinates
    :param scalar_values_array:  Nx1 array of scalar values to raster
    :param raster_meshgrid:      (optional) tuple of two arrays defining the coordinates of the grid for the raster
                                 If not defined, a grid covering all the extension of the cloud at the specified resolution
                                 if created  
    :param resolution:           (optional, required if raster_meshgrid i not defined
    :return: 
    '''
    # X, Y and S are just short names
    X = first_coord_array
    Y = second_coord_array
    S = scalar_values_array
    assert X.shape == Y.shape
    assert X.shape == S.shape

    # 2D CLOUD ------------------------------------------------------------------
    # cloud XY coordinates (Nx2)
    XY = np.vstack((X, Y)).T

    # GRID -------------------------------------------------------------------
    if raster_meshgrid is None:
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()

        # pixel centers
        x_range = np.arange(xmin + raster_resolution/2 , xmax, raster_resolution)
        y_range = np.arange(ymin + raster_resolution/2 , ymax, raster_resolution)

        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        raster_meshgrid = (X_mesh, Y_mesh)

        # grid XY coordinates (Nx2)
        XY_grid = np.vstack((X_mesh.ravel(), Y_mesh.ravel())).T
    else:
        # grid XY coordinates (Nx2)
        XY_grid = np.vstack((raster_meshgrid[0].ravel(), raster_meshgrid[1].ravel())).T

    # KD TREES ---------------------------------------------------------------
    print('Building kdtree...')
    cloud_tree = KDTree(XY)
    grid_tree = KDTree(XY_grid)

    print('Searching neighbors...')
    # search for the neighbors ----------
    indices_list = grid_tree.query_ball_tree(cloud_tree, search_radius)

    print('Reducing...')
    # build the raster -----------------------
    operation_function = getattr(np, operation)
    R = np.ones_like(raster_meshgrid[0]) * np.nan
    R_view = R.ravel()
    # for i in tqdm(range(len(indices_list))):
    #     indices = indices_list[i]
    #     if len(indices) > 0:
    #         R_view[i] = operation_function(S[indices])

    #---------------NUMBA-----------------
    # change the list of lists to a typed List of np.arrays
    L_indices_list = List( [np.array(l, dtype=int) for l in indices_list] )

    @jit(nopython=True)
    def operate(S, indices_list, R):
        R_view = R.ravel()
        for i in range(len(indices_list)):
            indices = indices_list[i]
            if len(indices) > 0:
                R_view[i] = operation_function(S[indices])
        return R
    # ---------------NUMBA-----------------

    R = operate(S, L_indices_list, R)

    return R

