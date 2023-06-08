import tensorflow as tf
from .customlayers import LassoRegression
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os


class LassomodPix_v2(tf.keras.Model):
    def __init__(self,data_path: str,):
        super(LassomodPix_v2, self).__init__()
        self._data_path = data_path
        self._smoke_chem_list =['SO2', 'PM2_5', 'NOx', 'VOCs', 'NH3', 'CO']
        projout = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'

        # 시도 정보 쉐이프파일
        ctp_rvn_gpd = gpd.GeoDataFrame.from_file(os.path.join(data_path,"geoinfodata","시도","ctp_rvn.shp",),encoding = 'cp949')
        ctp_rvn_gpd.crs = {'init':'epsg:5179'}
        self._ctp_rvn_gpd = ctp_rvn_gpd

       
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

        self.results_pd = pd.DataFrame(columns=['id','ctp_kor_nm','train_mse','test_mse','train_rmse','test_rmse','train_r2','test_r2','x','y'])
        self.grid_layers = []
        self.lose_mse = tf.keras.losses.MeanSquaredError()
        self.optimizers = []

        for pix_index in self._indexed_grid_point.index.tolist():
            
            self.grid_layers.append(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1(0.01)))
            # self.optimizers.append(tf.keras.optimizers.Adam(0.001))
    def call(self, inputs):
        pred_base = np.zeros([len(inputs),82,67,1])
        # for i, pix_index in enumerate(self._indexed_grid_point.loc[~self._indexed_grid_point.CTP_KOR_NM.isna()].index.tolist()):
        for i, pix_index in enumerate(self._indexed_grid_point.index.tolist()):
            x, y = pix_index%82,pix_index//82
            pred = self.grid_layers[i](inputs)
            pred_base[:,x,y,:] = pred
        return tf.convert_to_tensor(pred_base)
