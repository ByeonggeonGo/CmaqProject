# Deep learning based CMAQ surrogate model

In this study, we train deep learing based model for surrogate CMAQ(The Community Multiscale Air Quality Modeling System)

![https://github.com/SlownSteadi/CmaqProject/issues/1#issue-1754400603](https://user-images.githubusercontent.com/80737484/245418958-276fce13-7372-4e14-a590-9a52631c081e.png)
Image source:http://bioearth.wsu.edu/cmaq_model.html

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

<br>Among the 15 chemical species in CMAQ, only PM2.5 is used as the target.
<br>Since there are a total of 120 scenarios, the data described above is structured for each scenario individually.
