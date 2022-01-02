import pandas as pd
import numpy as np
from ccplot.hdf import HDF
from ccplot.algorithms import interp2d_12
import ccplot.utils
from shapely.geometry import Point, LineString, shape
import geopandas as gdp

gdp.options.display_precision = 16

def read_file(file_name, product_name=b'Perpendicular_Attenuated_Backscatter_532'):
    """
    file_name: .hdf calipso file
    returns calipso_datetime, depth, lats, lons, and dataset arrays from calipso file
    """
    
    with HDF(file_name) as product: 
        # Import datasets.
        calipso_datetime = product[b'Profile_UTC_Time'][:, 0]
        depth = product[b'metadata'][b'Lidar_Data_Altitudes']
        lats = product[b'Latitude'][:,0]
        lons = product[b'Longitude'][:,0]
        dataset = product[product_name][:]
    
    return calipso_datetime, depth, lats, lons, dataset

def get_index(target_lat, target_lon, target_depth, lons, lats, depth):
    """
    target_lat: start/stop lat. tuple
    target_lon: start/stop lon. tuple 
    target_depth: lower/upper depth in km. ex -0.04 is 40m below surface. 
    returns index based on lat, lon, and depth    
    """
    
    # find index based on lattitude and longitude
    lat_start = target_lat[0]
    lat_stop = target_lat[1]
    lon_start = target_lon[0]
    lon_stop = target_lon[1]
    
    lon_idx = np.where((lons >= lon_start) & (lons <= lon_stop))[0]
    lat_idx = np.where((lats >= lat_stop) & (lats <= lat_start))[0]
    
    # find min/max index for each lat/lon
    x1, x2 = lon_idx.min(), lon_idx.max()
    y1, y2 = lat_idx.min(), lat_idx.max()

    # lat-lon index range
    idx = (max(x1, y1), min(x2, y2))
    x1, x2 = idx

    # depth index
    h = np.where((depth >= target_depth[0]) & (depth <= target_depth[1]))[0]
    
    return idx, h

def get_index_depth(target_depth, lons, lats, depth):
    """
    target_lat: start/stop lat. tuple
    target_lon: start/stop lon. tuple 
    target_depth: lower/upper depth in km. ex -0.04 is 40m below surface. 
    returns index based on lat, lon, and depth    
    """

    # depth index
    h = np.where((depth >= target_depth[0]) & (depth <= target_depth[1]))[0]
    
    return h

def subset_data(filename, target_lat, target_lon, target_depth):
    """
    target_lat: start/stop lat. tuple
    target_lon: start/stop lon. tuple 
    target_depth: lower/upper depth in km. ex -0.04 is 40m below surface. 
    returns index based on lat, lon, and depth    
    """ 
    
    # read file and extract data
    calipso_datetime, depth, lats, lons, dataset = read_file(filename)
    
    # get index based on target lat/lon and depth
    idx, h = get_index(target_lat, target_lon, target_depth, lons, lats, depth)
    
    # subset data by index
    calipso_datetime = calipso_datetime[idx[0]:idx[1]]
    
    # Convert time to datetime.
    calipso_datetime = np.array([ccplot.utils.calipso_time2dt(t) for t in calipso_datetime])
    
    # subset lat/lon by index
    lats = lats[idx[0]:idx[1]]
    lons = lons[idx[0]:idx[1]]

    # dataset needs to be subsetted by index and depth
    dataset = dataset[idx[0]:idx[1], h]
    
    # subset depth by h
    depth = depth[h]
    
    return calipso_datetime, depth, lats, lons, dataset

def subset_depth(filename, target_depth):
    """
    target_lat: start/stop lat. tuple
    target_lon: start/stop lon. tuple 
    target_depth: lower/upper depth in km. ex -0.04 is 40m below surface. 
    returns index based on lat, lon, and depth    
    """ 
    
    # read file and extract data
    calipso_datetime, depth, lats, lons, dataset = read_file(filename)
    
    # get index based on target lat/lon and depth
    h = get_index_depth(target_depth, lons, lats, depth)
    
    # Convert time to datetime.
    calipso_datetime = np.array([ccplot.utils.calipso_time2dt(t) for t in calipso_datetime])

    # dataset needs to be subsetted by index and depth
    dataset = dataset[:, h]
    
    # subset depth by h
    depth = depth[h]
    
    return calipso_datetime, depth, lats, lons, dataset

def to_geoDataFrame(calipso_datetime, depth, lats, lons, dataset, 
                   bLineString=True):
    
    if bLineString:
        # create LineString segments for linear interpolation
        # zip lat/lon
        geom_list = [xy for xy in zip(lons, lats)]

        # create line segments between each lat/lon calipso point
        geom1 = [LineString([Point(x),Point(y)]) for x,y in zip(geom_list[:-1], geom_list[1:])]
        
        # index needs to be the same lenght in dataframe
        # last line in dataframe will be a point itself
        geom2 = [LineString([Point(geom_list[-1]),Point(geom_list[-1])])]
        
        geom = geom1 + geom2 
        
    else:
        geom = [Point(xy) for xy in zip(lats, lons)]
    
    # geopandas dataframe include lat, lon, geometry
    gdf = gdp.GeoDataFrame({'lat':lats, 'lon':lons, 'geometry':geom}, 
                                    crs='epsg:4326', geometry='geometry')
    
    # create temp df to join to geopandas dataframe
    vals = {'calipso_datetime': np.repeat(calipso_datetime, len(depth)),
           'lat': np.repeat(lats, len(depth)),
           'lon': np.repeat(lons, len(depth)),
           'depth': np.array(list(depth)*len(lats)).flatten(),
           'value':dataset.flatten(), 
           }
    
    df_temp = pd.DataFrame(vals)
    
    # join dataframes on lats/lons
    gdf = gdf.merge(df_temp)
    
    return gdf
    

def to_dataframe(calipso_datetime, depth, lats, lons, dataset):
    
    vals = {'calipso_datetime': np.repeat(calipso_datetime, len(depth)),
           'lat': np.repeat(lats, len(depth)),
           'lon': np.repeat(lons, len(depth)),
            'depth': np.array(list(depth)*len(lats)).flatten(),
            'value':dataset.flatten(), 
           }

    return pd.DataFrame(vals)
    
def visualize_backscatter(filename, target_lat, target_lon, target_depth, colormap, nz):
    """
    Visualize and save backscatter image based on target lat/lon/depth. 
    
    file_name: .hdf calipso file
    target_lat: start/stop lat. tuple
    target_lon: start/stop lon. tuple 
    target_depth: lower/upper depth in km. ex -0.04 is 40m below surface.  
    colormap: direcotry to calipso-backscatter.cmap
    nz: number of vertical pixels 
    """
    
    calipso_datetime, depth, lats, lons, dataset = subset_data(filename, target_lat, target_lon, target_depth)
    
    x1, x2 = latlon_idx
    h1, h2 = depth_range
    
    # Interpolate data on a regular grid.
    X = np.arange(x1, x2, dtype=np.float32)
    Z, null = np.meshgrid(height, X)
    data = interp2d_12(
        dataset[::],
        X.astype(np.float32),
        Z.astype(np.float32),
        x1, x2, x2 - x1,
        h2, h1, nz,
    )

    # Import colormap.
    cmap = ccplot.utils.cmap(colormap)
    cm = mpl.colors.ListedColormap(cmap['colors']/255.0)
    cm.set_under(cmap['under']/255.0)
    cm.set_over(cmap['over']/255.0)
    cm.set_bad(cmap['bad']/255.0)
    norm = mpl.colors.BoundaryNorm(cmap['bounds'], cm.N)

    # Plot figure.
    fig = plt.figure(figsize=(12, 6))
    
    im = plt.imshow(
                    data.T,
                    extent=(lats[0], lats[-1], h1, h2),
                    cmap=cm,
                    norm=norm,
                    aspect='auto',
                    interpolation='nearest'
    )
    
    ax = im.axes
    
    TIME_FORMAT = '%e %b %Y %H:%M:%S UTC'
    ax.set_title('CALIPSO %s - %s' % (
        calipso_datetime[0].strftime(TIME_FORMAT),
        calipso_datetime[-1].strftime(TIME_FORMAT)
    ))
    
    ax.set_xlabel('Lat')
    ax.set_ylabel('Altitude (km)')
    
    cbar = plt.colorbar(
        extend='both',
        use_gridspec=True
    )
    
    label = 'Perpendicular Attenuated Backscatter 532nm (km$^{-1}$ sr$^{-1}$)'
    cbar.set_label(label)
    fig.tight_layout()
    plt.savefig('calipso-plot.png')
    plt.show()
    
