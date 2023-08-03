import os
import sys

#s3 and external data connection libraries 
import boto3
import s3fs 
from s3fs.core import S3FileSystem
#import psycopg2
import pyarrow.parquet as pq
import getpass #to coonect to a database
import awswrangler as wr

#data analysis libraries
import pandas as pd
import numpy as np 
import xlrd 
import math
import random
#import pyreadr
from sklearn.cluster import DBSCAN

#plotting libraries 
import matplotlib.pyplot as plt
import seaborn as sns

#date time libraries
import datetime
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


#spatial libraries 
import fiona
import requests
from shapely.geometry import shape
from shapely import wkt
import json
from pandas.io.json import json_normalize
import shapely.wkt 
import folium 
import geopandas as gpd 
import fiona
from shapely.geometry import shape 
import folium 
from haversine import haversine, Unit
import pygeohash as pgh
from shapely import wkt

#geocoding libraries 
from geopy.geocoders import Nominatim
from ratelimit import limits,sleep_and_retry
from tqdm import tqdm

#python parallel processing libraries 
from functools import partial
from multiprocessing import Pool

#connect to s3 
s3=boto3.resource('s3')

''' This script contains frequently used functions. Please add to this list any of the functions that you find
    useful so that we all can use it (i.e. not reinvent the wheel). For example, calculating duration between 
    two time stamps or distance between two locations etc. Also, Document the steps here as well as in more 
    detail on confluence.'''    


def plot_tldf(in_df,col_sep,col_purp,col_weight,size_bin): 

    """
    This function computes the trip distributions by separation (could be travel time or distance)

    
    Usage
    -----
    out_df = plot_tldf(in_df,"travel_time","travel_purpose","travel_person_weight","size_bin")
    
    Parameters
    ----------
    
    out_df = output pandas dataframe 
    in_df = input pandas dataframe
    col_sep = column name representing the separation (travel time or distance) in the input dataframe (in_df)
    col_purp = column name that identifies the trip types in the input dataframe 
    col_weight = column name that reports the weight for each of the reported sample trip in the input dataframe 

    """
    
    
    #standard bins and labels
    #tl_bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    #tl_label = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    
    #user defined bins
    tl_bins=[]
    tl_label =[]
    val_bin=0
    max_bin = int(math.ceil(in_df[col_sep].max() / 10)) * 10
    
    for i in range(0,max_bin,size_bin):
        tl_bins.append(val_bin)
        val_bin=val_bin+size_bin
        #print(val_bin)
        tl_label = tl_bins[1:]
   
    #create bins
    txt_bin = "bin_"  + col_sep
    in_df.loc[:,txt_bin] = pd.cut(in_df[col_sep],bins=tl_bins,labels=tl_label) 
    
    print(len(col_purp))
    if len(col_purp) < 1:
        print("there is no purpose")
        col_purp = 'def_purp'
        in_df.loc[col_purp] = '0'
        
    if len(col_weight) < 1:
        print("there are no weights")
        col_weight = 'def_wgt'
        in_df.loc['def_wgt'] = '1'
    
    
    #compute sample and weighted trips by separation (travel time or distance, based on argument passed to function)
    tl_plt = in_df.groupby([txt_bin,col_purp]).agg({ col_weight: ['count','sum']}).reset_index()
    tl_plt.columns = ['tlen_separation','purp_des','sample_trips','weighted_trips']
    
    tl_plt.head()
    #function to compute shares within each subgroup 
    f_per = lambda x: 100 * x/float(x.sum())
    
    #computing shares within each subgroup for sample as well as weighted trips
    tl_plt[['per_sample','per_weighted']] = (tl_plt.groupby(['purp_des'])['sample_trips','weighted_trips'].transform(f_per))
   
    
    #out_df = in_df[[txt_bin,col_sep,col_purp]]
    out_df = tl_plt
    
    sns.set(style="white", palette="muted", color_codes=True)
    #xax_lbl=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    xax_lbl = tl_label
    
    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 1, figsize=(20,10), sharex=False)
    g=sns.lineplot(x="tlen_separation",y="per_sample",data=tl_plt,hue="purp_des",ax=axes[0])
    g.set_xticks(xax_lbl)
    g.set_xticklabels(xax_lbl,rotation=90)
    # Plot a kernel density estimate and rug plot
    sns.lineplot(x="tlen_separation",y="per_weighted",data=tl_plt,hue="purp_des",ax=axes[1])

    return(out_df)  


def hdist(col_frlat,col_frlng,col_tolat,col_tolng): 
    
    
    '''
    This function returns the haversine (strainght line) distance in miles between a pair of lat/longs from trip surveys 

    
    Usage
    -----
    df['dist'] = df.apply(lambda row: hdist(row.col_frlat,row.col_frlng,row.col_tolat,row.col_tolng),axis=1)
    
    Parameters
    ----------
    df = name of the dataframe
    df['dist'] = the new column being added to the dataframe with calculated distance
    col_frlat = name of the df column reporting the from latitude 
    col_frlng = name of the df column reporting the from longitude 
    col_tolat = name of the df column reporting the to latitude 
    col_tolng = name of the df column reporting the to longitude 
    '''

    #from_loc = (row[col_frlat],row[col_frlng])
    #to_loc = (row[col_tolat],row[col_tolng])
    from_loc = (col_frlat,col_frlng)
    to_loc = (col_tolat,col_tolng)
    dist_mi = haversine(from_loc,to_loc, unit='mi')
    return(dist_mi)

''' Geocoding function requires GeoPy, Nominatim,Rate limited, and TQDM Packages '''

def rate_limited_geocode(query):
    ''' delays the query to to limit 1 per second to geocode API'''
    return gc.geocode(query)

def geocode(row):
    '''
    This functions returns a point location (lat,long) for a given address. The row needs to include the following fields: street_address, city, postal_code
    
    Usage
    ---- 
    df['geo_location'] = df.apply(geocode,axis=1)
    '''
    lookup_query = row['street_address'] + "," + row["city"] + "," + row["region"] + "," + row["postal_code"]
    lookup_result = rate_limited_geocode(lookup_query)
    #time.sleep(2)
    #print(lookup_result)
    return(lookup_result)

''' Function to project a point pandas dataframe into points geodataframe'''
def proj_df(in_df,lat,lon):
    from shapely.geometry import Point 
    from geopandas.tools import sjoin

    ptgeom = [Point(xy) for xy in zip(in_df[lon],in_df[lat])]
   
    # Creating a Geographic data frame 
    crs = {'init': 'epsg:4326'}
    
    #generating a geodataframe with point as geometry feature 
    out_gdf = gpd.GeoDataFrame(in_df,crs=crs,geometry=ptgeom)
    return(out_gdf)

''' Function to tag a zone(TAZ id) to a point location or record in a dataframe'''
def tag_taz(in_df, indf_lat, indf_lon, geo_df, geo_taz, geo_geom):
    from shapely.geometry import Point
    from geopandas.tools import sjoin

    """creates a geom to tag taz within which it exists"""
    geom = [Point(xy) for xy in zip(in_df[indf_lon], in_df[indf_lat])]

    # Creating a Geographic data frame
    crs = {"init": "epsg:4326"}
    geo_df.crs = crs
    type(geo_df)

    # tagging TAZ using spatial join
    in_gdf = gpd.GeoDataFrame(in_df, crs=crs, geometry=geom)
    geo_df = gpd.GeoDataFrame(geo_df, crs=crs, geometry=geo_geom)
    out_gdf = gpd.sjoin(in_gdf, geo_df[[geo_taz, geo_geom]], how="left", op="within")

    return out_gdf


def create_map_layers(df,indf,col_lat,col_lon,col_cluster,layer1,layer2): 
    import matplotlib.cm as cm 
    #print(cm.hot(1))
    mean_lat = df[col_lat].mean()
    mean_lon = df[col_lon].mean()
    m = folium.Map(location=[mean_lat,mean_lon],zoom_start=9)
    
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.colors import Normalize    
   
    '''creating specific color for each cluster using dataframes for lookup (as each cluster is labeled randomly)'''
    colors_array = cm.rainbow(np.linspace(0,1, len(df[~df[col_cluster].isin(['-1'])][col_cluster].unique())))
    #colors_array = cm.rainbow(np.linspace(0,1,200)) # hardcoded range 
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    rainbow_lut = pd.DataFrame()
    rainbow_lut['cluster_label'] = df[~df[col_cluster].isin(['-1'])][col_cluster].unique()
    rainbow_lut['lut_cmap'] = rainbow
    rainbow_lut = rainbow_lut.reset_index()
    
    
    #print(colors_array)
    rainbow = [colors.rgb2hex(i) for i in colors_array]

    '''testing colormaps'''
    cmap = cm.autumn
    norm = Normalize(vmin=-20, vmax=10)
    cmap(norm(5))
    
    cluster_layer = folium.FeatureGroup(name=layer1)
    info_layer = folium.FeatureGroup(name=layer2)
    
   
    
    for _,row in df.iterrows(): 
        
        if row[col_cluster] == -1: #reads the value as number from dataframe
            cluster_color = '#000000'
        else: 
            cluster_id = row[col_cluster]
            cluster_color = rainbow_lut.loc[rainbow_lut['cluster_label'] == cluster_id]['lut_cmap'].tolist()[0] #rainbow[row[col_cluster]]
            
        cluster_layer.add_child(
            folium.CircleMarker(
            location= [row[col_lat],row[col_lon]],
            radius=5,
            popup= row['FromDate'] + ":" + str(row['FromTime'])+  ":" + str(row[col_cluster]),
            color=cluster_color,
            fill=True,
            fill_color=cluster_color
            )).add_to(m)
        m.add_child(cluster_layer)
    
    for _,row in indf.iterrows(): 
        info_layer.add_child(
            folium.Marker(
            location= [row['latitude'],row['longitude']],
            radius=5,
            popup= str(row['naics6']) + ":" + str(row['company']),
            #color=cluster_color,
            #fill=True,
            #fill_color=cluster_color
            )).add_to(m)
        m.add_child(info_layer)
    
    #cluster_layer.add_to(m)
    #info_layer.add_to(m)
    #folium.Layercontrol.add_to(m)
    m.add_child(folium.LayerControl())
    return(m)


''' Useful Resources
#### link 1: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
#### link 2: https://sites.google.com/site/python3tutorial/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
#### line 3: https://www.linkedin.com/pulse/parallel-apply-python-ian-lo '''

''' Used for pralleliztion: Example tagging TAZ for origin and destination trip ends'''
from functools import partial
from multiprocessing import Pool
def parallelize_dataframe(df,func,n_cores=5):
    df_split = np.array_split(df,n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func,df_split))
    pool.close()
    pool.join()
    return(df)


# +
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool,cpu_count


''' Used for parallelization: Example clustering function by group of records to an unique identifier '''
class WithExtraArgs(object):
    def __init__(self, func, **args):
        self.func = func
        self.args = args
    def __call__(self, df):
        return self.func(df, **self.args)


def applyParallel(dfGrouped, func, kwargs):
    p=3
    with Pool(cpu_count()) as p:
        ret_list = p.map(WithExtraArgs(func, **kwargs), [group for name, group in dfGrouped])
    return pd.concat(ret_list)


