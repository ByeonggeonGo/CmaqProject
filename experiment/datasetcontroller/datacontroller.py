from glob import glob
import os
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from copy import deepcopy
from rasterio.transform import from_origin
import rasterio
from rasterio.plot import show
import matplotlib
import tensorflow as tf

class DataController():
    def __init__(self, data_path: str,):
        self._data_path = data_path
        self._smoke_chem_list =['SO2', 'PM2_5', 'NOx', 'VOCs', 'NH3', 'CO']
        projout = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'

        # 시도 정보 쉐이프파일
        ctp_rvn_gpd = gpd.GeoDataFrame.from_file(os.path.join(data_path,"geoinfodata","시도","ctp_rvn.shp",),encoding = 'cp949')
        ctp_rvn_gpd.crs = {'init':'epsg:5179'}
        self._ctp_rvn_gpd = ctp_rvn_gpd

        cutting_polygon_coords = [(-180000, -550000), (-180000, 130000), (420000, 130000), (420000, -550000)]
        cutting_polygon = Polygon(cutting_polygon_coords)
        self._ctp_rvn_gpd_cutting = self._ctp_rvn_gpd.to_crs(projout).intersection(cutting_polygon)

        # 프로젝션 맞춰서 기본 그리드 형성
        # lcc프로젝션 기본 레스터 제작
        

        # out_grid = make_geocube(vector_data=test_data, measurements=["습도(%)"], geom=geom,resolution=(9000, 9000), fill=0, output_crs=projout) #for most crs negative comes first in resolution
        x_m = list(range(-180000,-180000 + 9000 * 67, 9000))
        y_m = list(range(-585000,-585000 + 9000 * 82, 9000))

        print(len(x_m), len(y_m))

        grid_points = []
        for x_i in x_m:
            for y_i in y_m:
                grid_points.append(Point(x_i,y_i))

        grid_data = pd.DataFrame(grid_points, columns=['geometry'])

        grid_data = gpd.GeoDataFrame(grid_data, geometry='geometry')
        grid_data.crs = ctp_rvn_gpd.to_crs(projout).crs
        grid_data.loc[:,'x_m'] = grid_data.geometry.x
        grid_data.loc[:,'y_m'] = grid_data.geometry.y
        grid_data.loc[:,'value'] = 0
        grid_data.loc[:,'index'] = grid_data.index

        joined = gpd.sjoin(ctp_rvn_gpd, grid_data.to_crs(5179), op='contains')
        indexed_grid_point = pd.merge(grid_data, joined.loc[:,['CTPRVN_CD', 'CTP_ENG_NM', 'CTP_KOR_NM', 'index_right']], how='left', left_on='index', right_on='index_right')
        indexed_grid_point = gpd.GeoDataFrame(indexed_grid_point, geometry='geometry')

        self._indexed_grid_point = indexed_grid_point

        # 연평균, 일평균 등등 모두 관리
        # path기반
        # path속 폴더들은 계층구조 동일하게 
    
    def get_ctp(self):
        return self._ctp_rvn_gpd
    
    def get_yearly_dataset(self,):
        avg_conc_path_list = glob(os.path.join(self._data_path,"yearly_data","concentration","*"))
        avg_conc_path_info_df = pd.DataFrame()
        avg_conc_path_info_df.loc[:,'path'] = avg_conc_path_list
        avg_conc_path_info_df.loc[:,'Run'] = [int(path.split("/")[-1].split(".")[1]) for path in avg_conc_path_list]

        avg_ems_path_list = glob(os.path.join(self._data_path,"yearly_data","emission","*"))
        avg_ems_path_info_df = pd.DataFrame()
        avg_ems_path_info_df.loc[:,'path_ems'] = avg_ems_path_list
        avg_ems_path_info_df.loc[:,'Run'] = [int(path.split("/")[-1].split(".")[1]) for path in avg_ems_path_list]

        cont_matrix = pd.read_csv(os.path.join(self._data_path,"02_Emission_0518.csv"))

        merged_info_df = pd.merge(avg_conc_path_info_df, avg_ems_path_info_df, how='left', on='Run')
        merged_info_df = pd.merge(merged_info_df, cont_matrix, how='left', on='Run')
        merged_info_df = merged_info_df.sort_values(by=['Run'])
        merged_info_df.index = range(len(merged_info_df))

        target_arr = np.concatenate([np.array(Dataset(path, 'r').variables['PM2_5'][0,0,:,:])[np.newaxis,:,:] for path in merged_info_df.path.tolist()])[:,:,:,None]

        smoke_list = [Dataset(path, 'r') for path in merged_info_df.path_ems.tolist()]
        smoke_val_list = []
        for i in self._smoke_chem_list:
            smoke_arr_i = np.concatenate([np.array(Dataset(path, 'r').variables[i][0,0,:,:])[np.newaxis,:,:] for path in merged_info_df.path_ems.tolist()])[:,:,:,None]
            smoke_val_list.append(smoke_arr_i)

        smoke_arr = np.concatenate(smoke_val_list, axis = 3)
        smoke_arr.shape, target_arr.shape
        input_cols = list(set(merged_info_df.columns) - set(['path','Run','path_ems']))

        control_matrix = merged_info_df.loc[:,input_cols].values
        self.input_cols = input_cols
        return control_matrix, smoke_arr, target_arr
    
    def check_grid_ctp_shp_map(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        self._indexed_grid_point.to_crs(5179).plot(ax = ax, markersize = 0.1)
        self._ctp_rvn_gpd.plot(ax = ax)

    
    def get_region_based_reshaped_smoke(self,smoke_arr):

        pd_reformed_list = []
        for run in range(len(smoke_arr)):
            ## 스모크 5채널 각 컬럼으로 값 넣어주기 (루프돌리면 가능할듯), ex: voc_value, sox_value, nox_value 등등...
            smoke_indexed_grid_point = deepcopy(self._indexed_grid_point)

            for i, pre_cor in enumerate(self._smoke_chem_list):
                sub_smoke_arr = smoke_arr[run][:,:,i]  #

                for pix_index in smoke_indexed_grid_point.index.tolist():
                    x, y = pix_index%82,pix_index//82
                    smoke_indexed_grid_point.loc[pix_index,pre_cor] = sub_smoke_arr[x,y]
            
            sub_pd = smoke_indexed_grid_point.loc[:,['CTP_KOR_NM','SO2','PM2_5','NOx','VOCs','NH3','CO']].groupby(['CTP_KOR_NM']).sum()

            sub_pd2 = sub_pd.stack().reset_index()
            sub_pd2.columns = ['CTP_KOR_NM', 'Pollutant', 'Value']

            sub_pd2['New_Column'] = sub_pd2['CTP_KOR_NM'] + '_' + sub_pd2['Pollutant']
            sub_pd2 = sub_pd2.drop(columns=['CTP_KOR_NM', 'Pollutant'])
            sub_pd2 = sub_pd2.iloc[:,[1,0]].T
            sub_pd2.columns = sub_pd2.iloc[0,:].values
            sub_pd2 = sub_pd2.iloc[1:2,:]
            sub_pd2.index = range(len(sub_pd2))

            pd_reformed_list.append(sub_pd2)

        r_smoke_dataset = pd.concat(pd_reformed_list, axis=0)
        r_smoke_dataset.index = range(len(r_smoke_dataset)+1)[1:]
        
        return r_smoke_dataset
    
    def region_based_smoke_to_grid_base(self, r_smoke_dataset,grid_weights):
        #
        r_smoke_to_smoke_shape_list = []

        for i in r_smoke_dataset.index.tolist():

            chem_base = np.zeros_like(grid_weights)
            for i_c, i_chem in enumerate(self._smoke_chem_list):

                
                for pix_index in self._indexed_grid_point.loc[~self._indexed_grid_point.CTP_KOR_NM.isna()].index.tolist():
                    x, y = pix_index%82,pix_index//82

                    region_str = self._indexed_grid_point.loc[pix_index,'CTP_KOR_NM']
                    sum_chem_value = r_smoke_dataset.loc[i, region_str + '_' + i_chem]  # i번시나리오 테스트
                    grid_weights_r_sum = r_smoke_dataset.loc[1, region_str + '_' + i_chem]  # 베이스 시나리오이므로 1번 고정
                    base_smkoe_r_i_value = grid_weights[x,y,i_c]

                    i_value = sum_chem_value * (base_smkoe_r_i_value / grid_weights_r_sum)

                    chem_base[x,y,i_c] = i_value
            
            r_smoke_to_smoke_shape_list.append(chem_base)
        
        return r_smoke_to_smoke_shape_list
    
    def check_grid_based_smoke(self, smoke_arr):
        smoke_indexed_grid_point = deepcopy(self._indexed_grid_point)

        for i, pre_cor in enumerate(self._smoke_chem_list):

            sub_smoke_arr = smoke_arr[:,:,i]

            for pix_index in smoke_indexed_grid_point.index.tolist():
                x, y = pix_index%82,pix_index//82
                smoke_indexed_grid_point.loc[pix_index,pre_cor] = sub_smoke_arr[x,y]

        for i,chem in enumerate(self._smoke_chem_list):
            plt.figure(figsize = [5,30])
            sub_ax = plt.subplot(6,1,i+1)
            plt.title(chem)
            self._ctp_rvn_gpd.plot(ax = sub_ax)
            smoke_indexed_grid_point.to_crs(5179).plot(ax = sub_ax,markersize = 1,column = chem)
    

    def get_ctp_based_pred_map(self, pred_arr):
      

        data = pred_arr 
        # 배열 크기
        height, width, bands = data.shape

        # 레스터 파일 생성
        dst_filename = os.path.join(self._data_path,"output.tif")  # 저장할 레스터 파일 경로와 이름 설정
        driver = 'GTiff'  # 레스터 파일 포맷 (여기서는 GeoTIFF로 설정)

        # 레스터 파일 매개변수 설정
        transform = from_origin(-180000, 153000, 9000, 9000)  # 좌표변환을 위한 transform 설정 (여기서는 임의로 설정)
        # transform = from_origin(153000, -180000, 9000, 9000)  # 좌표변환을 위한 transform 설정 (여기서는 임의로 설정)
        count = bands  # 밴드 개수
        dtype = data.dtype  # 데이터 타입

        # 레스터 파일 생성 및 데이터 쓰기
        with rasterio.open(dst_filename, 'w', driver=driver, height=height, width=width, count=count, dtype=dtype, transform=transform) as dst:
            for band in range(bands):
                dst.write(data[:, :, band], band + 1)
        
    
        with rasterio.open(dst_filename) as src:
            fig, ax = plt.subplots(figsize=(6, 6))
            data = src.read(1)
            # 컬러바
            image = ax.imshow(data, vmin=0, vmax=75)
            cbar = plt.colorbar(image, ax=ax,)

            # # 진짜 그리기
            image = show(src, ax=ax, vmin=0, vmax=75)
            self._ctp_rvn_gpd_cutting.plot(ax=ax, color='none', edgecolor='black', linewidth=0.2)
           


        return fig
    
    def _bytes_feature(self,value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _parse_function(self,tfrecord_serialized):
        features={
            'CMAQ_t': tf.io.FixedLenFeature([], tf.string),
            'SMOKE_t': tf.io.FixedLenFeature([], tf.string),
            'air_quality_monitoring_t': tf.io.FixedLenFeature([], tf.string),
            'weather_monitoring_t': tf.io.FixedLenFeature([], tf.string),
            'year': tf.io.FixedLenFeature([], tf.string),
            'month': tf.io.FixedLenFeature([], tf.string),
            'day': tf.io.FixedLenFeature([], tf.string),
            'hour': tf.io.FixedLenFeature([], tf.string),
                }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
        
        cmaq = tf.io.decode_raw(parsed_features['CMAQ_t'], tf.float32)
        smoke = tf.io.decode_raw(parsed_features['SMOKE_t'], tf.float32)
        aq = tf.io.decode_raw(parsed_features['air_quality_monitoring_t'], tf.float64)  #이것 둘 64로 저장됨
        weather = tf.io.decode_raw(parsed_features['weather_monitoring_t'], tf.float64)
        year = tf.io.decode_raw(parsed_features['year'], tf.uint8)
        month = tf.io.decode_raw(parsed_features['month'], tf.uint8)
        day = tf.io.decode_raw(parsed_features['day'], tf.uint8)
        hour = tf.io.decode_raw(parsed_features['hour'], tf.uint8)

    

        cmaq = tf.reshape(cmaq, [82, 67, 1])
        
        smoke = tf.reshape(smoke, [82, 67, 45])
      
        aq = tf.reshape(aq, [82, 67, 1])
     
        weather = tf.reshape(weather, [82, 67, 15])
      
        
        year = tf.squeeze(year)
        month = tf.squeeze(month)
        day = tf.squeeze(day)
        hour = tf.squeeze(hour)

        
        # image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        # image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])


        # classes = tf.io.decode_raw(parsed_features['classes'], tf.uint8)    
        # classes = tf.squeeze(classes)
        tf.cast(cmaq,tf.float32)

        return cmaq, smoke, aq, weather, year, month, day, hour
    
    def _parse_function2(self,tfrecord_serialized):
        features={
            'CMAQ_t': tf.io.FixedLenFeature([], tf.string),
            'SMOKE_t': tf.io.FixedLenFeature([], tf.string),
            'air_quality_monitoring_t': tf.io.FixedLenFeature([], tf.string),
            'weather_monitoring_t': tf.io.FixedLenFeature([], tf.string),
            # 'year': tf.io.FixedLenFeature([], tf.string),
            # 'month': tf.io.FixedLenFeature([], tf.string),
            # 'day': tf.io.FixedLenFeature([], tf.string),
            # 'hour': tf.io.FixedLenFeature([], tf.string),
                }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
        
        cmaq = tf.io.decode_raw(parsed_features['CMAQ_t'], tf.float32)
        # 12년 측정망자료로 나중에 정규화 해주기
        smoke = tf.io.decode_raw(parsed_features['SMOKE_t'], tf.float32)
        aq = tf.io.decode_raw(parsed_features['air_quality_monitoring_t'], tf.float64)  #이것 둘 64로 저장됨
        # 바람 방향 x, y벡터랑 강도로 뽑는작업 나중에 하기
        weather = tf.io.decode_raw(parsed_features['weather_monitoring_t'], tf.float64)
        # year = tf.io.decode_raw(parsed_features['year'], tf.uint8)
        # 월 sin함수 주기로 바꾸어서 넣어주기
        # month = tf.io.decode_raw(parsed_features['month'], tf.uint8)
        # 일 sin함수 주기로 바꾸어서 넣어주기
        # day = tf.io.decode_raw(parsed_features['day'], tf.uint8)
        # hour = tf.io.decode_raw(parsed_features['hour'], tf.uint8)

    

        cmaq = tf.reshape(cmaq, [82, 67, 1])
        # print(cmaq.shape)
        smoke = tf.reshape(smoke, [82, 67, 45])
        # print(smoke.shape)
        aq = tf.reshape(aq, [82, 67, 1])
        # # print(aq.shape)
        weather = tf.reshape(weather, [82, 67, 15])
        # print(weather.shape)
        
        # year = tf.squeeze(year)
        # month = tf.squeeze(month)
        # day = tf.squeeze(day)
        # hour = tf.squeeze(hour)

        
        # image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        # image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])


        # classes = tf.io.decode_raw(parsed_features['classes'], tf.uint8)    
        # classes = tf.squeeze(classes)
        # tf.cast(cmaq,tf.float32)
        # tf.concat([tf.cast(smoke,tf.float32),tf.cast(aq,tf.float32),tf.cast(weather,tf.float32),], axis=3)
        # tf.concat([tf.cast(smoke,tf.float32),tf.cast(aq,tf.float32),tf.cast(weather,tf.float32),], axis=2)[8:-10,2:-1,:]
        # return cmaq[8:-10,2:-1,:], tf.cast(smoke,tf.float32)[8:-10,2:-1,:], year, month, day, hour
        # return smoke[8:-10,2:-1,:],cmaq[8:-10,2:-1,:]
        # return smoke,cmaq
        return tf.concat([tf.cast(smoke,tf.float32),tf.cast(aq,tf.float32),tf.cast(weather,tf.float32),], axis=2)[8:-10,2:-1,:], cmaq[8:-10,2:-1,:]
    
    def get_window_dataset(self, record_path_list: list, window_size: int, shift: int, batch_size: int):
        def get_target_window(x1,x2):
            return x2.batch(window_size, drop_remainder=True)

        def get_input_window(x1,x2):
            return x1.batch(window_size, drop_remainder=True)

        def dense_step_input(input_batch):
        # Shift features and labels one step relative to each other.
            return input_batch[:-1]

        def dense_step_target(target_batch):
        # Shift features and labels one step relative to each other.
            return target_batch[-1:]
        dataset = tf.data.TFRecordDataset(record_path_list)
        dataset = dataset.map(self._parse_function2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).window(window_size, shift=shift, drop_remainder = True)

        dataset_cmaq = dataset.flat_map(get_target_window)
        target_cmaq = dataset_cmaq.map(dense_step_target)

        dataset_smoke = dataset.flat_map(get_input_window)
        input_smoke = dataset_smoke.map(dense_step_input)
        
        ds = tf.data.Dataset.zip((input_smoke, target_cmaq)).batch(batch_size).shuffle(1000)
        
        return ds, dataset



    
    

        