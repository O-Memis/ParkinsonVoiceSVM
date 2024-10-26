# ParkinsonVoiceSVM
 
 "Parkinson's Disease Detection using Machine Learning - Python"
 Project of classification of Parkinsons's  disease with dataset of  voice features. Supervised classification is done by using Support Vector Machine model from Sci-kit learn library.



## Content
1) Files
2) Dataset and video tutorial
3) How to reference
4) Information about the project
5) Code Organization



## 1-Files

Code file is **"ParkinsonVoiceSVM.py"** and dataset file is **"parkinsons.csv"** you should download these two.
There is also a model file called **"svm_parkinson.joblib"** to import the model in a system.



## 2-Dataset and video tutorial

Utilized by this video tutorial : [Siddardhan](https://youtu.be/HbyN_ey-JVc?si=0SMd4H0ybejVNkVx) 
Codes in tutorial are **changed & re-designed based on my own decisions and practices.**



## 3-How to reference

Please refer with names or Github links which are Siddardhan S (tutorial owner), and Oğuzhan Memiş (this repository), and the source of the dataset.
Contacts are welcomed.

[Dataset](https://www.kaggle.com/datasets/thecansin/parkinsons-data-set)
Oxford Parkinson's Disease Detection Dataset    
2008-06-26
Cite: Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008),'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease',IEEE Transactions on Biomedical Engineering



## 4-Information about the project

Parkinson's is a neurological and progressive disease, which occurs from lack of some neurotransmitters in the brain such as Dopamine.
It harms the fine-tuned control of the movements of the body. As it progress, several symptoms such as tremor, body stiffness and loss of balance are observed. 

Dataset: 
    Instances (samples): 195
    Attributes (features): 22 + 1(label)
 
Focused on various **acoustic properties of voice recordings**, which are used to detect and analyze Parkinson's disease.

Voice measurements from 31 people, 23 with Parkinson's disease (PD).6 recordings per patient.
The main aim is discriminate **healthy=0 and PD=1** as binary classification.

How these measurements are performed:

**Voice Recording:** Subjects are typically asked to sustain a vowel sound (often 'ahhh') for several seconds.
**Signal Processing:** The voice recordings are then processed using specialized software that can extract these acoustic features.
**MDVP:** Many of these features are extracted using **Multi-Dimensional Voice Program (MDVP)**, a software tool for voice analysis.
**Advanced Analysis:** Some of the more complex measures (like RPDE, D2, DFA) require advanced mathematical analysis of the voice signal.



# 5-Code Organization

   The codes are separated into 7 different cells based on 5 steps.
    Read the descriptions and run the codes cell by cell. You can also run the whole code at once, if you want.
    
    The steps are as follows:
        1) Importing the data
        2) EDA
        3) Preprocessing by 3 options. **Run 1 of them and compare the results.** 
        4) Model training and tuning
        5) Model usage


Cells are constructed by this command: **#%%** in Spyder IDE.


