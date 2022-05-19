import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from scipy.spatial import KDTree
from tqdm import tqdm
import sys
from plyfile import PlyData, PlyElement
from glob import glob

sys.path.append('.') # needed for pdal to see the module
import cloud_raster



def test_loading_with_plyfile():

    filename_globs=[
        "/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_JAX/s2p/JAX_214/JAX_214_006_PAN_CROPPED_JAX_214_007_PAN_CROPPED/tiles/*/*/cloud.ply",
        "/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_JAX/s2p/JAX_214/JAX_214_006_PAN_CROPPED_JAX_214_008_PAN_CROPPED/tiles/*/*/cloud.ply",
        "/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_JAX/s2p/JAX_214/JAX_214_006_PAN_CROPPED_JAX_214_009_PAN_CROPPED/tiles/*/*/cloud.ply",
    ]

    X = []
    Y = []
    Z = []
    for g in filename_globs:
        filenames = glob(filename_globs[0])
        for f in filenames:
            plydata = PlyData.read(f)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            print(x.shape, y.shape, z.shape)
            X.extend(list(x))
            Y.extend(list(y))
            Z.extend(list(z))

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    print('#points =', X.shape[0])

    test_compute_raster(X, Y, Z, search_radius=0.5, raster_resolution=0.5, operation='nanmax') #nanmean, nanmin, nanmax, etc.


def pdal_test_compute_raster(ins, outs):
    # Function to be called by a pdal pipeline
    # Coomand line:  pdal pipeline pdal_pipeline.json
    # See: https://pdal.io/index.html
    # conda install -c conda-forge pdal python-pdal gdal

    X = ins['X']
    Y = ins['Y']
    Z = ins['Z']

    test_compute_raster(X,Y,Z,search_radius=1, raster_resolution=0.5)
    return True


def simple_test_compute_raster():
    # BUILD A CLOUD FROM A DSM
    gt_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/gt/JAX_214_DSM.tif'
    gt = imread(gt_filename)

    y_size, x_size = gt.shape
    raster_resolution = 0.5
    x_origin = 442345.345  # invented
    y_origin = 1231313.7890 # invented

    x_range = np.arange(x_origin, x_origin + raster_resolution * x_size, raster_resolution)
    y_range = np.arange(y_origin, y_origin + raster_resolution * y_size, raster_resolution)

    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    U_mesh, V_mesh = np.meshgrid(np.arange(x_size,dtype=int), np.arange(y_size, dtype=int))

    X = X_mesh.ravel()
    Y = Y_mesh.ravel()
    Z = gt[V_mesh.ravel().astype(int), U_mesh.ravel().astype(int)].ravel()

    # TODO: add noisy points


    # a search_radius of 2 will smooth things
    test_compute_raster(X,Y,Z, search_radius=2, raster_resolution=0.5)

    return True


def test_compute_raster(X, Y, Z, search_radius=0.5, raster_resolution=0.5, operation='nanmedian'):


    R = cloud_raster.compute_raster(X, Y, Z, search_radius=search_radius, raster_resolution=raster_resolution, operation=operation)

    plt.figure(400)
    vmin, vmax = np.nanpercentile(R, (2, 98))
    #plt.imshow(R, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imshow(R, cmap='terrain')

    plt.colorbar()
    title=f'search_radius {search_radius} raster_resolution {raster_resolution} operation {operation}'
    plt.title(title)
    imsave(f'{title}.tif', R)
    plt.show()



if __name__ == "__main__":
    test_loading_with_plyfile()
    #simple_test_compute_raster()