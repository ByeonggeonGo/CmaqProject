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
