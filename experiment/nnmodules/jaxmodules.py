import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import grad, jit, vmap
from jax import random
import optax
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np


class Jax_lasso_cmaq():

    def __init__(self, alpha: float,data_path: str):
        self.alpha = alpha
        

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
        
    def get_vmap_ds(self, X,y):
        input_ds = [X for i in range(82*67)]
        target_ds = []
        for i, pix_index in enumerate(self._indexed_grid_point.index.tolist()):
            x_cor, y_cor = pix_index%82,pix_index//82
            sub_target = y[:,x_cor,y_cor,:]
            target_ds.append(sub_target)
        
        return input_ds, target_ds


    def train_map(self, X, y, learning_rate_init, epochs):

        input_ds, target_ds = self.get_vmap_ds(X,y)

        @jax.jit
        def lasso_loss(params, x_batched, y_batched):
            
            model  = nn.Dense(features=1, kernel_init=nn.initializers.xavier_uniform())
            def squared_error(x, y):
                pred = model.apply(params, x)
                return jnp.inner(y-pred, y-pred) / 2.0
       
            mse_loss = jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
            l1_loss = jnp.sum(jnp.abs(params['params']['kernel']))

            # Vectorize the previous to compute the average of the loss on all samples.
            return self.alpha * l1_loss + mse_loss
        
        def train(input, target):
            loss_grad_fn = jax.value_and_grad(lasso_loss)
            model  = nn.Dense(features=1, kernel_init=nn.initializers.xavier_uniform())
            key = random.PRNGKey(0)
            input_dim = input.shape[1]
            output_dim = 1
            params = model.init(key, jnp.ones((1, input_dim)))

            lr = learning_rate_init
            tx = optax.adam(learning_rate=lr)
            opt_state = tx.init(params)
            for i in range(epochs):
                loss_val, grads = loss_grad_fn(params, input, target)
                updates, opt_state = tx.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            
            return params
        
        loss_grad_fn = jax.value_and_grad(lasso_loss)
        results = jax.vmap(train)(jnp.array(input_ds), jnp.array(target_ds))
        self.all_params = results

    def predict(self, inputs, batch_size):
        origin_inputs = inputs

        quotient = len(inputs) // batch_size
        remainder = len(inputs) % batch_size
        loop_n = quotient + 1 if remainder !=0 else quotient

        batch_results = []
        for i in range(loop_n):
            if i != (loop_n-1):
                inputs = origin_inputs[batch_size*i:batch_size*(i+1)]
            else:
                inputs = origin_inputs[batch_size*i:]

            input_ds = [inputs for i in range(82*67)]
            kernels_weigts_ds = self.all_params['params']['kernel']
            bias_weigts_ds = self.all_params['params']['bias']

            model  = nn.Dense(features=1, kernel_init=nn.initializers.xavier_uniform())
            weighs_dic = {
                'params': {
                    'kernel':[],
                    'bias':[],
                }
            }

            def predict(input, kernel_weights, bias_weights):
                weighs_dic['params']['kernel'] = kernel_weights
                weighs_dic['params']['bias'] = bias_weights
                pred = model.apply(weighs_dic, input)
                return pred
            
            def pred_reshape(grid_index, grid_pred):
                x_cor, y_cor = grid_index%82,grid_index//82
                pred_base = jnp.zeros([len(grid_pred),82,67,1])
                # pred_base[:,x_cor,y_cor,:] = grid_pred
                pred_base = pred_base.at[:,x_cor,y_cor,:].set(grid_pred)
                return pred_base
            
            results = jax.vmap(predict)(jnp.array(input_ds), jnp.array(kernels_weigts_ds), jnp.array(bias_weigts_ds))

            grid_index_list = self._indexed_grid_point.index.tolist()
            pred_test_reshape = jax.vmap(pred_reshape)(jnp.array(np.array(grid_index_list).reshape(-1,1)), jnp.array(results.reshape(5494,len(inputs),1,1)))
            result_final = jnp.sum(pred_test_reshape, axis = 0)
            batch_results.append(result_final)
        
        return np.concatenate(batch_results)


