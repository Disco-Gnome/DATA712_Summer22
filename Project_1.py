### Set up our environment
import os
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.impute import SimpleImputer
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib as mp
import matplotlib.pyplot as plt

#The following is just to enable TensorFlow with GPU acceleration.
#I was originally working on a much larger dataset that needed this.
#I switched datasets due to overflow problems, but I kept this code.
import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\cudart64_110.dll")
import tensorflow as tf

###Load & format our dataset
os.chdir("D:\School\Summer 2022\Advanced Data Analysis")
GAMMA = pd.read_csv("magic04.data", header=None)
GAMMA.columns = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','gamma']

#I would use LabelEncoder() here, but that encodes h=1 & g=0,
#with no simple method for remapping. I want g=1 and h=0 explicitly,
#so I create this column manually
GAMMA['gamma_enc'] = GAMMA['gamma']
GAMMA['gamma_enc'].replace({'g':"1", 'h':"0"},
                           inplace=True)
GAMMA['gamma_enc'] = GAMMA['gamma_enc'].astype(int)
GAMMA = GAMMA.drop('gamma', axis=1)

GAMMA.info()
GAMMA.describe()

###Create Training & Testing sets
GAMMA_X = GAMMA.drop('gamma_enc', axis=1)
GAMMA_y = GAMMA['gamma_enc']
X_train, X_test, y_train, y_test = train_test_split(GAMMA_X,
                                                    GAMMA_y,
                                                    stratify=GAMMA_y)

###Exploring our training set
X_train.info()
X_train.describe()

y_train.info()
y_train.describe()

###Confirm no missing values and clean if necessary
GAMMA.isnull().any()
#There are no missing values in our total dataset, and by extension
#there are none in our training & testing sets.
#Additionally, the cleaning and formatting done as part of prepping
#our dataset earlier appears to have been enough for the data to be
#ready for next steps.

###Visualization
y_train.plot(kind='hist',
             alpha=0.8,
             title="Gamma Particles vs Hadrons",
             grid=False,
             figsize=(4,4),
             fontsize=14,
             bins=3)
plt.ylabel("Observations")
plt.xticks([0.16, 0.84], ["Hadrons", "Gamma Particles"])
plt.show()

#Next, I would use some variation of: pd.plotting.scatter_matrix(X_train)
#But a default scatter matrix comes out looking crowded, with lots of
#overlapping text & irrelevant units. So, I do the following to adjust
#the appearance and remove sub-axis tics & labels.
sm = pd.plotting.scatter_matrix(X_train,
                                diagonal='kde',
                                figsize=(10,10),
                                alpha=0.2)
for subaxis in sm:
    for ax in subaxis:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
X_train_scatter = sm[0][0].get_figure()
#scatter_matrix.savefig("ScatterMatrix.png")
del (ax, subaxis, sm)

###Transform Data
#Note that this code is just to demonstrate competence with
#transformations. The variables given in our dataset are all
#already in a specific form for analysis. Several are
#themselves already transformations or derivations commonly
#used in this type of astronomy (called "Hillas parameters").
#I may have to leave this code unused in future projects.
#I save all measurements as part of a new dataframe to keep
#them separate. If I were planning to actually use these,
#I would append these as new columns on my full data.

#So, for example, if we wanted to log-normalize our ellipse
#Length or Width, which have very high variation. We would:
transform_df = pd.DataFrame()
transform_df['fLength'] = GAMMA['fLength']
transform_df['fLength_log'] = np.log(GAMMA["fLength"])
print("fLength Variance:", np.var(GAMMA['fLength']))
print("Log-fLength Variance:", np.var(transform_df['fLength_log']))
#Comparing distributions before and after log-normalization
GAMMA['fLength'].plot(kind='hist',
                      title="Distribution Before Normalization")
plt.ylabel('Frequency')
plt.xlabel('fLength in mm')
plt.show()
transform_df['fLength_log'].plot(kind='hist',
                                 title="Distribution After Normalization")
plt.ylabel('Frequency')
plt.xlabel('log fLength in mm')
plt.show()

#For Width, we would have to implement a solution for the existence
#of 0 values if we want to log-normalize. One solution would be to
#impute using mean value. SimpleImputer only allows imputation of
#NA values, so here I impute zeros to mean manually
transform_df['fWidth'] = GAMMA['fWidth']
fWidth_mean = np.mean(transform_df['fWidth'])
transform_df['fWidth'] = transform_df['fWidth'].replace(0, fWidth_mean)
transform_df['fWidth_log'] = np.log(transform_df['fWidth'])
print("fWidth Variance:", np.var(transform_df['fWidth']))
print("Log-fWidth Variance:", np.var(transform_df['fWidth_log']))

#Comparing distributions before and after log-normalization
GAMMA['fWidth'].plot(kind='hist',
                      title="Distribution Before Normalization")
plt.ylabel('Frequency')
plt.xlabel('fWidth in mm')
plt.show()
transform_df['fWidth_log'].plot(kind='hist',
                                 title="Distribution After Normalization")
plt.ylabel('Frequency')
plt.xlabel('log fWidth in mm')
plt.show()

#If we wanted to normalize using squaring, cubing, or exponentials instead,
#we might want to look at variables that have a wider distribution, like
#fConc and fConc1. These are measures of the relative brightness of the
#brightest pixel(s) in a given observation.

#Squaring
plt.hist(GAMMA['fConc']**2, bins=10)
plt.hist(GAMMA['fConc1']**2, bins=10)

#Cubing
plt.hist(GAMMA['fConc']**3, bins=10)
plt.hist(GAMMA['fConc1']**3, bins=10)

#Exponentials
plt.hist(np.exp(GAMMA['fConc']), bins=10)
plt.hist(np.exp(GAMMA['fConc1']), bins=10)

#Scatter matrix comparisons
attributes=['fConc', 'fConc1']
#Original values
scatter_matrix(GAMMA[attributes], figsize=(10,10))
#Squared values
scatter_matrix(GAMMA[attributes]**2, figsize=(10,10))
#Cubed Values
scatter_matrix(GAMMA[attributes]**3, figsize=(10,10))
#Exponentials
scatter_matrix(np.exp(GAMMA[attributes]), figsize=(10,10))


