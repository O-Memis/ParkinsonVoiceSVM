
"""
Jul 26  2024


"Parkinson's Disease Detection using Machine Learning - Python"
Utilized by this video tutorial : https://youtu.be/HbyN_ey-JVc?si=0SMd4H0ybejVNkVx 
Codes are changed & rearranged based on my own decisions and practices.



Dataset: https://www.kaggle.com/datasets/thecansin/parkinsons-data-set
    Will be also shared with the repository.
    

    
Reference: Please refer with names or Github links which about Siddardhan S, Oğuzhan Memiş (Oguzhan Memis), and the source of the dataset.
Contacts are welcomed.



CODE ORGANIZATION:
    The codes are separated into 7 different cells based on 5 steps.
    Read the descriptions and run the codes cell by cell. You can also run the whole code at once, if you want.
    
    The steps are as follows:
        1) Importing the data
        2) EDA
        3) Preprocessing by 3 options. Run 1 of them and compare the results.
        4) Model training and tuning
        5) Model usage
"""

#%% 1) Importings and Data

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale , StandardScaler
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

'''
KEY INFORMATION ABOUT THE DATASET
__________________________________
    
Oxford Parkinson's Disease Detection Dataset    
2008-06-26
Cite: Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008),'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease',IEEE Transactions on Biomedical Engineering


Parkinson's is a neurological and progressive disease, which occurs from lack of some neurotransmitters in the brain such as Dopamine.
It harms the fine-tuned control of the movements of the body. As it progress, several symptoms such as tremor, body stiffness and loss of balance are observed. 

Dataset: 
    Instances (samples): 195
    Attributes (features): 22 + 1(label)
 
Focused on various acoustic properties of voice recordings, which are used to detect and analyze Parkinson's disease.

Voice measurements from 31 people, 23 with Parkinson's disease (PD).6 recordings per patient.
The main aim is discriminate healthy=0 PD=1 as binary classification.

How these measurements are performed:

Voice Recording: Subjects are typically asked to sustain a vowel sound (often 'ahhh') for several seconds.
Signal Processing: The voice recordings are then processed using specialized software that can extract these acoustic features.
MDVP: Many of these features are extracted using Multi-Dimensional Voice Program (MDVP), a software tool for voice analysis.
Advanced Analysis: Some of the more complex measures (like RPDE, D2, DFA) require advanced mathematical analysis of the voice signal.


22 features:
    MDVP:Fo(Hz) --Average frequency of vocal fold vibration during speech. Typically 85-180 Hz for males, 165-255 Hz for females.
    MDVP:Fhi(Hz) -- The highest frequency of vocal fold vibration,Reduced in some PD. Usually under 500 Hz.  
    MDVP:Flo(Hz) --Minimum vocal fundamental frequency.Typically above 50 Hz.
    MDVP:Jitter(%) --Variation of fundamental frequency, expressed as a percentage. Typically <1% in healthy voices, can be higher in PD which is voice instability.
    MDVP:Jitter(Abs) --Usually in microseconds, typically <83.2 μs for healthy voices.
    MDVP:RAP --Short-term (cycle-to-cycle) irregularity measure. Typically <0.680% in healthy voices.
    MDVP:PPQ -- Another measure of frequency irregularity over a slightly longer term. Typically <0.840% in healthy voices.
    Jitter:DDP --Difference of Differences of Periods,variability of the fundamental frequency. Similar range.
    MDVP:Shimmer --Variability of the peak-to-peak amplitude in percentage. Typically <3.810% in healthy , often increased in PD.
    MDVP:Shimmer(dB)-- In a logarithmic scale. Typically <0.350 dB in healthy.
    Shimmer:APQ3 --Short-term (3-point) amplitude variability measure. Typically <3.070% in healthy.
    Shimmer:APQ5 -- Similar to APQ3, but uses 5 points instead of 3. Typically <4.230% in healthy.
    MDVP:APQ --Longer-term amplitude variability. Typically <3.810% in healthy.
    Shimmer:DDA --Similar range to other shimmer measures, typically low in healthy voices.
    NHR: Noise-to-Harmonics Ratio : Amount of noise in the signal. Higher in PD due to increased breathiness. Typically <0.190 in healthy.
    HNR: Harmonics-to-Noise Ratio : Inverse of NHR, measures voice clarity. Lower in PD. Typically >20 dB in healthy.
    RPDE: Recurrence Period Density Entropy -- Quantifies the repetitiveness of a time series, predictability of vocal fold vibrations. 0 to 1, higher values indicating more complexity.
    D2: Correlation Dimension --  Measure of signal complexity. Can detect subtle nonlinear changes in voice. Typically between 1 and 3.
    DFA: Detrended Fluctuation Analysis -- Statistical self-similarity of a signal at different time scales. Typically between 0.5 and 1, with 0.5 indicating uncorrelated noise.
    spread1 -- Nonlinear measure of fundamental frequency variation. Higher values indicate more variation.
    spread2 -- Complements spread1 in measuring F0 (center, or fundamental frequency) variability.
    PPE: Pitch Period Entropy -- Impredictability of pitch periods. Typically between 0 and 1, with higher values indicating unpredictability, as PD patients have.

'''

pdata = pd.read_csv("parkinsons.csv")
# Importing errors can be solved in different way. If you encounter problems, try change your path variable or the folder which code runs in it.


pd.set_option('display.float_format', '{:.6f}'.format)
pdata.head() # further observations can be done in EDA step.



y = pdata.iloc[:, [17]].to_numpy() # labels are located in 17. column


x = pdata.iloc[:,1:24].to_numpy()
x = np.delete(x, 16, axis=1) # labels are deleted



print(str(np.sum(y)))  # 147 of 195 are the Parkinson's. Dataset is unbalanced.

'''
The issue of unbalanced data can be solved by various methods such as downsampling(rejection), augmentation techniques, GANs etc.  
'''


#%% 2) Exploratory Data Analysis

pdata.info()  # Getting basic information about columns and numbers of non-null data



fulldescribe= pd.DataFrame()
fulldescribe = pdata.describe()  # To observe the basic statistics


# Plot1 : Statistics of dataset
plt.figure(figsize=(16, 12))
sns.heatmap(fulldescribe.transpose(), annot= True, cmap="Purples", fmt=".2f")  # numeric format can be changed.
plt.show()





# Obtain the rows including the label we want to see
status = pdata.sort_values(by="status", ascending= False )  # Sort for the status column to see the differences.

parkinsons = pdata[pdata['status'] == 1] # Filter rows where status is 1
healthies = pdata[pdata['status'] == 0]





# Plot 2: Comparison between labels
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10)) # to make a subplot.
sns.heatmap(parkinsons.describe().transpose(), annot= True, cmap="Purples", fmt=".2f", ax=axes[0])
axes[0].set_title('Parkinsons')

sns.heatmap(healthies.describe().transpose(), annot= True, cmap="Purples", fmt=".2f", ax=axes[1])
axes[1].set_title('Healthy ')
plt.tight_layout()
plt.show()





# Plot3 : Correlation between the features
numerical_data = pdata.select_dtypes(include=[float, int])  # Select only numerical columns

correlations = numerical_data.corr() # Shows correlation among the columns, which columns are similar

plt.figure(figsize=(16, 12))
sns.heatmap(correlations, annot= True, cmap="Purples")
plt.show()




'''
We can clearly see that, scale of features varies. In simpler terms, some features have much bigger numbers than others.
Also, some features are negatively correlated.

'''


#%% 3.a) Dataset with Normalization by fitting to all

# Reshape the array to shape (195,) to help Scikit models work in more compatible way
y = y.reshape(195,)


'''
If some features are much bigger than other, it will reduce the overall score. 
It is manipulating the data by applying Z-score normalization which assumes Normal Distribution for the data
It helps in comparing features that have different units or scales.

'''



scale = StandardScaler()

scale.fit(x) # Adapting to our data
x_scaled = scale.transform(x) # Apply the transformation to obtain the data in new scales.


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,test_size=0.4 , random_state=42)



#%% 3.b) Dataset with Normalization by fitting just to training


'''
Fitting the scaler on both training and test data introduces data leakage. 
This means that the model can indirectly "seen" the influence of test data on training data. The model has access to small amount of information from the test set, during training.
By only using the training statistics, you ensure that the model is evaluated just based on how well it can adapt to unseen data.
Therefore, fitting process can be setted just by using training set, to transform both training and test sets.
'''

y = y.reshape(195,)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4 , random_state=42)


scale = StandardScaler()

scale.fit(x_train) # Adapting to just training set

x_train = scale.transform(x_train) # Transform
x_test = scale.transform(x_test) 


#%% 3.c) Dataset without Normalization


y = y.reshape(195,)



# Bigger scales of data, can make the Kernel (dimensionality increase) process much longer.
# Especially the Polynomial Kernel takes enormously long time.

# You can can try to scale down some features. It can effect in a good or in a bad way.



'''
x[:, 0] = x[:, 0]/1000
x[:, 1] = x[:, 1]/1000
x[:, 2] = x[:, 2]/1000
x[:, 15] = x[:, 15]/10
x[:, 18] = x[:, 18]/10
'''


# Train-test split with randomization
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4 , random_state=42)



# DO NOT RUN THEM IN SAME TIME, INSTEAD RUN 1 OF THESE 3 SECTIONS TO SEE THE DIFFERENCE


#%% 4) SVM with GridSearchCV 

'''

Support Vector Machine is a frequently used algorithm for both classification and regression tasks.

Main idea is, finding the optimal plane which linearly separates the datapoints in feature space.
Feature Engineering can be done by the Kernel functions, to invent more features about the data. 
In order to increase the dimension, and find a way to separate the classes linearly.

In the algorithm, the datapoints are tried to separate by the decision boundary, which is a plane between the datapoints.
Support vectors are the other planes which represents closest points of each class .
Distance between the support vectors, including the decision boundary, is called the margin.
In higher dimensions, these planes are called "hyperplanes". Distance between the planes are calculated by using Normal vectors.
SVM aims to find the decision boundary that maximizes the margin between the classes.
The algorithm tries to increase this margin by using distance calculations and changing the hyperplane, in order to optimize the classification result. 
This is achieved through an optimization process that involves solving a convex optimization problem.

Hyperparameters:
-Kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid kernels. To map the data into linearly separable shape.
-C value tells algorithm how much avoid the misclassifications. High C value can cause overfitting, low C value can cause underfitting,
 optimal C value adds required regularization.
-Gamma value tells what number of points used to determine the boundary. 
 low value includes farther points, high value only includes few closest points.
'''


mymodel = svm.SVC(probability=True) # Classifier model

mymodel.get_params()  # Look for which parameters can be choosen




'''
Grid search is about simply iterating all sets of values defined, to obtain best performed model.
It also uses Cross-Validation. And parameter grid need to be defined.
'''


# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.6, 1, 8, 9, 10, 11, 12, 13],
    'kernel': ['linear', 'rbf' , 'sigmoid' ],
    'gamma': ['scale', 'auto',  0.01, 0.1, 0.25, 0.3, 0.33, 0.4, 0.5, 0.7,  1, 2, 10 , 15]
    
    }


grid_search = GridSearchCV(mymodel, param_grid, cv=4, n_jobs=-1, verbose=0)
# Setting n_jobs to -1 means that the algorithm will use all available CPU cores on your machine
# Verbose setting is for observing the details of the ongoing process 

grid_search.fit(x_train, y_train)



# Getting the best model
best_model = grid_search.best_estimator_




# Calculate the metrics for classification 

predictions = best_model.predict(x_test) # Make predictions

s1 = accuracy_score(y_test, predictions)
s2 = f1_score(y_test, predictions, average='weighted')
s3 = recall_score(y_test, predictions, average='weighted')
s4 = precision_score(y_test, predictions)

print(f"Accuracy: {s1*100:.2f}%")
print(f"F1 Score: {s2*100:.2f}%")
print(f"Recall: {s3*100:.2f}%")
print(f"Precision: {s4*100:.2f}%")


# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Heatmap of confusion matrix
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# Print the best model parameters
print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")


'''
Without normalization:
    Best parameters:
   C: 1
   gamma: scale
   kernel: linear
   85.90%
   
   
With standard normalization:
    Best parameters:
    C: 8 or 10
    gamma: 0.25 or 0.3 
    kernel: rbf
    93.59%
'''




#%% 5) Prediction system with importing


# For real-life usage of this model, the model can be saved in special formats.
# Then the model file can be imported and used for individual predictions on new data.


# Example prediction system with our current data as follows:
  
    
from joblib import dump, load


# Saving 
model_file_path = "svm_parkinson.joblib"
dump(best_model, model_file_path)
print(f"Model saved to {model_file_path}")



# Importing
modell = load(model_file_path)


# Select a row. Do not use to apply your transformations steps to the selected data as well.
select = int(input("Please enter a number to choose a row in dataset : "))

prediction2 = modell.predict(scale.transform(x[select,:].reshape(1,-1)))  






# Print the result
print(prediction2)

if prediction2 == 1 :
    print("Parkinson")
elif prediction2 == 0 :
    print("Healthy")


# Testing our result
if prediction2 == y[select] :
    print("The prediction is true")
else :
    print("The prediction is wrong")    
