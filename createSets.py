
import h5py
import numpy as np
from datafilereader import DataFileReader
from sklearn.utils import shuffle
import pickle

enForeign = False

#load data from group 5
folder = "./data/Groupe5/dataSetAGF-bobine3/"

labels = list(range(0,8))
labels_name = ["unknown","5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
featureListL = [4,5,6,7,8,9,10,12,61]


#get frequecy of the first coin (always the same)
dataset = DataFileReader(folder+labels_name[1]+".h5")
f,_ = dataset.get_all_mesurements()

#create a DataSet with all the measurements of all the coins
X = []
Y = []

folders = ["./data/Groupe5/dataSetAGF-bobine3/","./data/Groupe11/", "./data/Groupe10/"]
for j in range(len(folders)):
    for i in range(1, 8):
        dataset = DataFileReader(folders[j]+labels_name[i]+".h5")

        #get all measured Z for this coin
        _,Z = dataset.get_all_mesurements()
        N = len(Z) #number of measurements for this coin
        R = np.real(Z)
        L = np.imag(Z)/(2*np.pi*f)

        # substract all the data by the calibration
        R = R[1:,:]-R[0,:]
        L = L[1:,:]-L[0,:]

        #extract the needed features
        R = R[:,featureListR]
        L = L[:,featureListL]
        #d = (L[6]-L[5])/(f[10]-f[9])
        p = R[:,0:9]/L
        #concatenate the features
        for n in range(N-1):
            X.append(np.concatenate((R[n,:],L[n,:],p[n,:]),axis=0))
            #X.append(d) ,p[n,:]
            Y.append(i)
# if enForeign:
#     # add unknown coins on the validation set and on the training set
#     folder = "./data/foreign/"

#     labels_name = ["1_EUR","1_LST","1ct_EUR","2ct_EUR","2ct_LST","5ct_EUR", "5ct_LST_1991", "5ct_LST_2015", "10ct_EUR", "10ct_LST", "20ct_EUR", "20ct_LST","50ct_LST"]

#     featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
#     featureListL = [4,5,6,7,8,9,12,61]

#     dataset = DataFileReader(folder+labels_name[1]+".h5")
#     f,_ = dataset.get_all_mesurements()


#     for i in range(1, len(labels_name)):
#         dataset = DataFileReader(folder+labels_name[i]+".h5")

#         #get all measured Z for this coin
#         _,Z = dataset.get_all_mesurements()
#         N = len(Z) #number of measurements for this coin
#         R = np.real(Z)
#         L = np.imag(Z)/(2*np.pi*f)

#         # substract all the data by the calibration
#         R = R[1:,:]-R[0,:]
#         L = L[1:,:]-L[0,:]

#         #extract the needed features
#         R = R[:,featureListR]
#         L = L[:,featureListL]

#         #concatenate the features
#         for n in range(N-1):
#             X.append(np.concatenate((R[n,:],L[n,:]),axis=0))
#             Y.append(0)
            
print("number of data :",len(Y))
print("number of features :",len(X[0]))
print("number of different classes :",len(set(Y)))

#shuffle the data
X,Y = shuffle(X,Y,random_state=42)

#save the data in the different sets
N = len(Y)
Ntrain = int(0.6*N)
Nvalid = int(0.2*N)
Ntest = N-Ntrain-Nvalid


# Save the training set
with open("dataset/trainingset.pkl","wb+") as f:
    pickle.dump((X[:Ntrain],Y[:Ntrain]), f)
# Save the validation set
with open("dataset/validationset.pkl","wb+") as f:
    pickle.dump((X[Ntrain:Ntrain+Nvalid],Y[Ntrain:Ntrain+Nvalid]), f)
# Save the test set
with open("dataset/testingset.pkl","wb+") as f:
    pickle.dump((X[Ntrain+Nvalid:],Y[Ntrain+Nvalid:]), f)



