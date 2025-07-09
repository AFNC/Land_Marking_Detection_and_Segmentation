import os
import ocr
import glob
import folium
import geopandas as gpd
from osgeo import ogr
from shapely.geometry import Point, Polygon


def geopackage_with_meta_info(image_path, model_path):
    
    directory = 'geopackage_files'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    save_path = './geopackage_files/'
    coordinates_list, reference_numbers_list = ocr.ref_num_extraction(image_path, model_path)

    for i in range(len(coordinates_list)):
        polygon = Polygon(coordinates_list[i])
        gdf = gpd.GeoDataFrame(crs = {'init' :'epsg:4326'})
        gdf.loc[0,'name'] = 'land_marking'
        gdf.loc[0, 'geometry'] = polygon
        file_name = 'segmented_land'+str(i)+'.gpkg'
        file_path = os.path.join(save_path, file_name)         
        gdf.to_file(file_path, driver="GPKG")
        
    path = './geopackage_files/'
    gpkg_list = list(glob.glob(path+'*.gpkg'))
    for i in range(len(gpkg_list)):
        myfile = ogr.GetDriverByName('GPKG').Open(path+gpkg_list[i].split('\\')[-1], 1)
        reference_nums = {}
        reference_nums['Reference No. for segmented land'+str(i)] = str(reference_numbers_list[i])
        myfile.SetMetadata(reference_nums)
        myfile=None
        myfile = ogr.GetDriverByName('GPKG').Open(path+gpkg_list[i].split('\\')[-1], 1)
        print(myfile.GetMetadata() )
        myfile=None

        
        
        
image_path = 'stockton_1.png'
model_path = 'runs/segment/train15/weights/last.pt'
geopackage_with_meta_info(image_path, model_path)