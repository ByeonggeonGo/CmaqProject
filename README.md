# Deep learning based CMAQ surrogate model

In this study, we train deep learing based model for surrogate CMAQ(The Community Multiscale Air Quality Modeling System)

![https://github.com/SlownSteadi/CmaqProject/issues/1#issue-1754400603](https://user-images.githubusercontent.com/80737484/245418958-276fce13-7372-4e14-a590-9a52631c081e.png)
Image source:http://bioearth.wsu.edu/cmaq_model.html


* Purpose
<br> The objective of this study is to replace the computationally expensive CMAQ modeling with a deep learning-based surrogate model. 
<br> This approach aims to reduce the computational cost and enable the utilization of air quality prediction and simulation testing. 
<br> By using the surrogate model, which is based on deep learning, the research seeks to achieve cost reduction and facilitate various applications related to air quality prediction and simulation testing.

# Dataset

![smokecmaq](https://github.com/SlownSteadi/CmaqProject/assets/80737484/7f92388f-2881-4648-8cf4-70edb889f262)
<br>The dataset will be kept confidential for security reasons.
<br>The dataset consists of three main components: Control matrix, SMOKE model output, and CMAQ model output.
<br>The structure involves using the values from the control matrix as inputs for the SMOKE model, and the resulting output is then fed into the CMAQ model for air quality modeling.
<br>The control matrix is composed of 120 scenarios, which were used to model PM2.5 air quality for one year.
In total, there are 120 sets of scenarios (Control matrix -> SMOKE -> CMAQ)

## data shape

<br>The original data is structured as netCDF files, and each individual scenario is composed as follows:

* Control matrix: 119 (119 characteristics of pollutant emission sources, 17 regions * 7 emission sectors)
* SMOKE: 4 months (1, 4, 7, 11) * 30 days * 24 hours * 82 (x-axis grid) * 67 (y-axis grid) * 17 (z-axis grid) * 61 (chemical species)
* CMAQ: 4 months (1, 4, 7, 11) * 30 days * 24 hours * 82 (x-axis grid) * 67 (y-axis grid) * 1 (z-axis grid) * 15 (chemical species)

<br>Among the 15 chemicals in CMAQ, only PM2.5 is used as the target.
<br>Since there are a total of 120 scenarios, the data described above is structured for each scenario individually.

# Algorithms
<br> The surrogate model created in this research serves two main purposes:


* Original time-series data:
<br> The surrogate model is designed to capture the characteristics and patterns of the original hourly spatio-temporal data. It aims to provide a computationally efficient alternative to the original data while preserving its essential information and temporal aspects.

* yearly averaged data: 
<br> The surrogate model is developed to generate year level averaged PM2.5 Concentrations. By considering different scenarios, the model can provide averaged PM2.5 concentrations that represent the air quality conditions under specific conditions or scenarios. This allows for the analysis and comparison of air quality across different scenarios without the need for extensive computations.

![flowchart1](https://github.com/SlownSteadi/CmaqProject/assets/80737484/f1d10e9e-9528-42fe-acc5-32e648c65506)

* surrogate target 1
<br> In surrogate target 1, a model is created to predict CMAQ using SMOKE data, which is composed of hourly data. The length of the entire time window is determined based on the desired prediction timeframe for PM2.5. The time lag of the input data is a hyperparameter, but it is recommended to minimize it as much as possible while achieving good prediction performance. The target value can be set by adjusting the shift according to the specific objective of the model.

* surrogate target 2
<br> In surrogate target 2, a model is created to predict CMAQ directly from the control matrix as input. Since it involves average data, the characteristics of time windows and shifts, which are typical for time-series models, are not considered. The purpose of this model is to directly predict the average values of CMAQ from the control matrix, thereby reducing the cost of SMOKE modeling. From a policy perspective, using the control matrix as input allows for the establishment of clear emission reduction goals. This approach offers the advantage of setting precise emission reduction targets when utilizing the control matrix.

* surrogate target 3
<br> In surrogate target 3, a model is created to predict the average CMAQ from the average SMOKE as input. Since it involves average data, the characteristics of time windows and shifts, which are typical for time-series models, are not considered. The purpose of this model is to reduce the cost of CMAQ by predicting the average values of CMAQ. Considering that running a one-year simulation in CMAQ takes approximately one month, utilizing average simulations can achieve significant cost savings.

<br> Additionally, in surrogate target 3, there is an additional step of generating SMOKE data to be used as inputs. This is done through a generation algorithm to create average SMOKE data that achieves the target PM2.5 concentration. The purpose of this process is to find the optimal SMOKE configuration that can meet the desired PM2.5 target. By incorporating a generation algorithm, the research includes the generation of average SMOKE data as part of the surrogate target 3 process.

# experiments

* surrogate target 1

<br> for surrogate target 1 model, see experiment/smoke_cmaq_hourly.ipynb file.
<br> The LSTM-CNN based model and conditional DCGAN based model(pix2pix structure) were applied in the research, and the conditional DCGAN based model demonstrated superior performance.
<br> for model code see experiment/nnmodules/unet Lstm2dUnet class and experiment/nnmodules/gan DCGAN_v2 class

![lstmbase](https://github.com/SlownSteadi/CmaqProject/assets/80737484/922cdf01-7c86-4c2a-83b2-b76d166e3659)

lstm-cnn based model

![dcgan base](https://github.com/SlownSteadi/CmaqProject/assets/80737484/24953440-587e-430f-8133-0fcfbbd990ec)

conditional dcgan based model 

* surrogate target 2

<br> for surrogate target 2 model, see experiment/controlmatrix_cmaq_y.ipynb file.
<br> for model code see experiment/nnmodules/unet Unet_v3, Unet_v4,Unet_v5,
<br> Among the three classes, the Unet_v5 class model was found to have the least occurrence of overfitting when comparing the training set and the test set. In addition, the Unet_v5 class model also provides an analysis tool for further analysis.

![Unet_v5_structure drawio](https://github.com/SlownSteadi/CmaqProject/assets/80737484/3f704c84-3dbe-4fc2-a83d-76294ca86ff9)
--v5 model structure flowchart--

<br>The trained model provides the following functionalities:

<br>PM2.5 average concentration simulation based on input values: Utilizing the model, you can simulate the PM2.5 average concentration based on input values. This allows you to obtain predicted results for specific input conditions.

<br>Saving predicted values and input values: The model can generate predicted PM2.5 average concentration values for the given input. These predicted values can be saved along with the corresponding input values. This enables you to store the model's predictions and input data for future analysis or visualization purposes.

<br>Analysis tool integration: The Unet_v5 class model comes with an analysis tool. This tool allows you to evaluate the performance of the model, visualize simulation results, and conduct further analysis. It may include additional statistical analysis or visualization features, and can be used to perform various analyses based on different input values.

<br>With these functionalities, the trained model enables simulation of PM2.5 average concentrations based on input values, saving of predicted and input values, and provides an analysis tool for further analysis.

![dashboard_1](https://github.com/SlownSteadi/CmaqProject/assets/80737484/d90d5399-7d82-4a27-8c3d-64594eff2fdc)
<analysis tool image 1>
![dashboard_2](https://github.com/SlownSteadi/CmaqProject/assets/80737484/a23811ce-54fd-47aa-bc51-ff6e6300b9f1)
<analysis tool image 2>

<br>contribute:
<br>UI and backend developer: ~!@#!@#! url
<br>model development: url
<br>GIS engineering:



* surrogate target 3

<br> for surrogate target 3 model, see experiment/smoke_cmaq_model.ipynb file.
<br> for model code see experiment/nnmodules/unet Unet_v2






