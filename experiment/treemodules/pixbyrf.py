
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import seaborn as sns

class RFmodPix():
    def __init__(self, data_path: str,):
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
        self.models = []
    
    def fit(self, X_train,X_test, y_train, y_test):

        
        for pix_index in self._indexed_grid_point.loc[~self._indexed_grid_point.CTP_KOR_NM.isna()].index.tolist():
            x, y = pix_index%82,pix_index//82
            y_train_sub = y_train[:,x,y,:]
            y_test_sub = y_test[:,x,y,:]

            rand_reg = RandomForestRegressor(random_state=42, max_depth=3, n_estimators=50)
            rand_reg.fit(X_train, y_train_sub.ravel())
            self.models.append(rand_reg)
        
            y_pred_train = rand_reg.predict(X_train)
            y_pred = rand_reg.predict(X_test)

            train_mse = mean_squared_error(y_train_sub, y_pred_train, squared= True)
            test_mse = mean_squared_error(y_test_sub, y_pred, squared= True)

            train_rmse = mean_squared_error(y_train_sub, y_pred_train, squared= False)
            test_rmse = mean_squared_error(y_test_sub, y_pred, squared= False)

            train_r2 = r2_score(y_train_sub, y_pred_train,)
            test_r2 = r2_score(y_test_sub, y_pred,)
            

            i_ctp_nm = self._indexed_grid_point.CTP_KOR_NM.values[pix_index]

            self.results_pd.loc[pix_index,:] = [pix_index,i_ctp_nm,train_mse,test_mse,train_rmse,test_rmse,train_r2,test_r2,x,y]
        

        self.results_pd.loc[:,'model'] = self.models

        return 
    
    def predict(self, ctp_kor: str, inputs):
        pred_base = np.zeros([len(inputs),82,67,1])

        if ctp_kor != '전국':

            for ind in self.results_pd.loc[self.results_pd.ctp_kor_nm == ctp_kor].index.tolist():
                ind_val = self.results_pd.loc[ind,:].values
                pred = ind_val[10].predict(inputs)

                pred_base[:,ind_val[8],ind_val[9],0] = pred
            
        else:
            for ind in self.results_pd.index.tolist():
                ind_val = self.results_pd.loc[ind,:].values
                pred = ind_val[10].predict(inputs)

                pred_base[:,ind_val[8],ind_val[9],0] = pred


        return pred_base
    
    def get_ctp_kor_list(self,) -> list:
        kor_names = list(set(self.results_pd.ctp_kor_nm.values.tolist()))
        return kor_names
    
    def get_performance_map(self, ctp_kor: str, is_test:bool, index_name: str):
        # index_name: r2, rmse, mse

        perf_map = np.zeros([82,67,1])

        if ctp_kor != '전국':
            for ind in self.results_pd.loc[self.results_pd.ctp_kor_nm == ctp_kor].index.tolist():
                
                if is_test:
                    ind_val = self.results_pd.loc[ind,['x', 'y', f'test_{index_name}']].values
                else:
                    ind_val = self.results_pd.loc[ind,['x', 'y', f'train_{index_name}']].values
                perf_map[ind_val[0],ind_val[1],0] = ind_val[2]
        else:
            for ind in self.results_pd.index.tolist():
                
                if is_test:
                    ind_val = self.results_pd.loc[ind,['x', 'y', f'test_{index_name}']].values
                else:
                    ind_val = self.results_pd.loc[ind,['x', 'y', f'train_{index_name}']].values
                perf_map[ind_val[0],ind_val[1],0] = ind_val[2]

        return perf_map
    
    def one_one_plot(self, pred, true):
        plt.plot(pred[pred != 0], true[pred != 0], 'bo', markersize = 1)

        return
    
    def get_shap_summary_plot(self, X: np.array ,col_names, ctp_kor: str, ctp_eng: str, fig_size: list):

        plt.figure(figsize = fig_size)
        plt.title(ctp_eng)
        shap_base = np.zeros([X.shape[0],82,67,119])

        if ctp_kor != '전국':
            for ind in self.results_pd.loc[self.results_pd.ctp_kor_nm == ctp_kor].index.tolist():
                ind_val = self.results_pd.loc[ind,:].values
                # pred = ind_val[10].predict(X_test)
                shap_values = shap.TreeExplainer(ind_val[10]).shap_values(X)
                shap_base[:,ind_val[8],ind_val[9],:] = shap_values
        
        else:
            for ind in self.results_pd.index.tolist():
                ind_val = self.results_pd.loc[ind,:].values
                # pred = ind_val[10].predict(X_test)
                shap_values = shap.TreeExplainer(ind_val[10]).shap_values(X)
                shap_base[:,ind_val[8],ind_val[9],:] = shap_values

        df_loop_list = []
        for i in range(82*67):
            df_loop_list.append(X)

        reshape_input = np.concatenate(df_loop_list)
        shap.summary_plot(shap_base.reshape(-1,119), reshape_input, feature_names=col_names,show=False,max_display=20)

        
        # return shap_base
    
    def get_shap_map(self, X: np.array ,col_names, fig_size: list, save_path:str):

       
        shap_base = np.zeros([X.shape[0],82,67,119])

        for ind in self.results_pd.index.tolist():
            ind_val = self.results_pd.loc[ind,:].values
            # pred = ind_val[10].predict(X_test)
            shap_values = shap.TreeExplainer(ind_val[10]).shap_values(X)
            shap_base[:,ind_val[8],ind_val[9],:] = shap_values
        
        for i, col_name in enumerate(col_names):
            plt.figure(figsize = fig_size)
            plt.title(col_name)
            sns.heatmap(shap_base[0,:,:,i][::-1])

            plt.savefig(os.path.join(save_path,col_name.split(")")[1]))