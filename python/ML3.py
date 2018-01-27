
# coding: utf-8

import mnist
from mnist import show

# In[2]:


import os
import struct
import numpy as np
    
def ascii_show(image):
    """
    Render the array corresponding to the image.
    """
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print(row)


# ### Data representation

# In[3]:


training_data = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_LIST_OF_PAIR)[0:1000]
testing_data = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_LIST_OF_PAIR)[0:100]
print(len(training_data))
pixels, label = training_data[24]
print(label)
print(pixels.shape)
show(pixels)
ascii_show(pixels)


# In[4]:


# Create two lists, one containing the pixels, one containing the labels

# Training lists
Ptrain=[]
ytrain=[]
# Testing lists
Ptest=[]
ytest=[]
for i in range(0,len(training_data)):
    pixels, label = training_data[i]
    Ptrain.append(pixels)
    ytrain.append(label)
for i in range(0,len(testing_data)):
    pixels, label = testing_data[i]
    Ptest.append(pixels)
    ytest.append(label)


# ### Features :

# In[5]:


# Count the number of black pixels for each image (pixel value > 250)

def blackDigits(P):
    result=[]
    for i in range(0,len(P)):
        pixels=P[i]
        count=0
        for j in range(0,28):
            for k in range(0,28):
                if (pixels[j][k]>250):
                    count+=1
        result.append(count)
    return result


# In[6]:


# Detection of horizontal and vertical lines
# Contains two lists : 
# MaxBlackDigits_row is the list of the largest row of black pixels for each image 
# MaxBlackDigits_col is the list of the largest column of black pixels for each image

def MaxBlackDigits(P):
    MaxBlackDigits_row=[]
    MaxBlackDigits_col=[]
    for i in range(0,len(P)):
        pixels=P[i]
        res_row=0
        res_col=0
        for j in range(0,28):
            nj_row=0
            nj_col=0
            for k in range(0,28):
                if (pixels[j][k]>250):
                    nj_row+=1
                if (pixels[k][j]>250):
                    nj_col+=1    
            if nj_row>res_row:
                res_row=nj_row
            if nj_col>res_col:
                res_row=nj_col
        MaxBlackDigits_row.append(res_row)
        MaxBlackDigits_col.append(res_col)
    return [MaxBlackDigits_row,MaxBlackDigits_col]


# In[7]:


# Loop detection (first shot, will be improved...)
# Return three lists
# The first list contains the number of loops
# The second list contains the size of the loops, ie the number of pixels under the loops
# The third list contains the size of caves ie the number of pixels under curves

def loopIsolation(M,i,j):
    if M[i][j]<150:
        M[i][j]=250
        loopIsolation(M,i,j+1)
        loopIsolation(M,i+1,j)
        loopIsolation(M,i,j-1)
        loopIsolation(M,i-1,j)

def isInCave(M,m,n):
    if M[m][n]>0:
        return False
    s=0
    count=0;
    for i in range(0,m):
        s+=M[i][n]
    if s==0:
        count+=1
    s=0
    for i in range(m+1,len(M)):
        s+=M[i][n]
    if s==0:
        count+=1
    s=0
    for i in range(0,n):
        s+=M[m][i]
    if s==0:
        count+=1
    s=0
    for i in range(n+1,len(M[0])):
        s+=M[m][i]
    if s==0:
        count+=1
    if count==1:
        return True

def numberOfLoops(L):
    result=0
    for i in range(0,len(L)-1):
        if (L[i]==1 and L[i+1]==0):
            result+=1
    return result

def LoopsAndCavesDetection(P):
    nbLoops=[]
    nbDigitsInLoops=[]
    nbDigitsInCaves=[]
    for i in range(0,len(P)):
        pixels=P[i]
        N=np.array(np.zeros((28,28),dtype='i'))
        for i in range(0,28):
            for j in range(0,28):
                N[i][j]=pixels[i][j]
        for i in range(0,28):
            N[0][i]=250
            N[27][i]=250
            N[i][0]=250
            N[i][27]=250
        loopIsolation(N,1,1)
        nL=0
        nC=0
        RowDigitsInLoop=np.zeros(28)
        for j in range(0,28):
            for k in range(0,28):
                if N[j][k]<50:
                    nL+=1
                    RowDigitsInLoop[j]=1
                if isInCave(pixels,j,k):
                    nC+=1
        nbLoops.append(numberOfLoops(RowDigitsInLoop))
        nbDigitsInLoops.append(nL)
        nbDigitsInCaves.append(nC)
    return [nbLoops,nbDigitsInLoops, nbDigitsInCaves]


# In[8]:


#Illustrates loop detection :
M=Ptrain[17]
N=np.array(np.zeros((28,28),dtype='i'))
for i in range(0,28):
    for j in range(0,28):
        N[i][j]=int(M[i][j])
for i in range(0,28):
    N[0][i]=int(250)
    N[27][i]=int(250)
    N[i][0]=int(250)
    N[i][27]=int(250)
loopIsolation(N,1,1)
ascii_show(M)
ascii_show(N)


# ### Predictions with scikit-learn :

# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors #k-nearest neighboors
from sklearn import linear_model
import time


# In[10]:


# Calculate features for the training set
beg=time.time()

f1=blackDigits(Ptrain[0:10000])
f23=MaxBlackDigits(Ptrain[0:10000])
f2=f23[0]
f3=f23[1]
f456=LoopsAndCavesDetection(Ptrain[0:10000])
f4=f456[0]
f5=f456[1]
f6=f456[2]

end=time.time()
print(int(end-beg))


# In[11]:


# Concatenate the features into Xtrain
Xtrain=np.vstack((np.array(f1),np.array(f2),np.array(f3),np.array(f4),np.array(f5),np.array(f6))).T


# In[12]:


# Calculate features for the testing set
g1=blackDigits(Ptest)
g23=MaxBlackDigits(Ptest)
g2=g23[0]
g3=g23[1]
g456=LoopsAndCavesDetection(Ptest)
g4=g456[0]
g5=g456[1]
g6=g456[2]


# In[13]:


# Concatenate the features into Xtest
Xtest=np.vstack((np.array(g1),np.array(g2),np.array(g3),np.array(g4),np.array(g5),np.array(g6))).T


# In[14]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
Xtrain = sc_x.fit_transform(Xtrain)
Xtest = sc_x.transform(Xtest)


# In[15]:


# Knn predictions 
knn = neighbors.KNeighborsClassifier()
knn.fit(Xtrain, ytrain[0:10000])
y_predict_knn = knn.predict(Xtest)


# In[16]:


# Linear regression 
lin = linear_model.SGDClassifier()
lin.fit (Xtrain, ytrain[0:10000])
y_predict_lin = lin.predict(Xtest)


# In[17]:


# Gradient boosting predictions 
gbc = GradientBoostingClassifier()
gbc.fit(Xtrain, ytrain[0:10000])
y_predict_gbc = gbc.predict(Xtest)


# In[18]:


# Random Forests predictions 
rf = RandomForestClassifier(n_estimators=10)
rf.fit(Xtrain, ytrain[0:10000])
y_predict_rf = rf.predict(Xtest)


# ### Accuracy of the predictions :

# In[19]:


# Pourcentage of success with knn
count=0
for i in range(0,len(y_predict_knn)):
    if y_predict_knn[i]==ytest[i]:
        count+=1
print(100*count/len(y_predict_knn))


# In[20]:


# Pourcentage of success with linear model
count=0
for i in range(0,len(y_predict_lin)):
    if y_predict_lin[i]==ytest[i]:
        count+=1
print(100*count/len(y_predict_lin))


# In[21]:


# Pourcentage of success with gradient boosting
count=0
for i in range(0,len(y_predict_gbc)):
    if y_predict_gbc[i]==ytest[i]:
        count+=1
print(100*count/len(y_predict_gbc))


# In[22]:


# Pourcentage of success with random forests
count=0
for i in range(0,len(y_predict_rf)):
    if y_predict_rf[i]==ytest[i]:
        count+=1
print(100*count/len(y_predict_rf))


# In[23]:


# Confusion Matrix : 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_predict_gbc,)
cm

# Visualize confusion matrix as a heatmap
import seaborn as sb
plt.rcParams['figure.figsize'] = (10, 7)
ax = sb.heatmap(cm, cmap="BuPu")
ax.invert_yaxis()
plt.yticks(rotation=0);

# In[24]:


# Representation of predictions for each digits with random forests method :
import matplotlib.pyplot as plt
x=range(0,10)
L=np.zeros((10,10))
for i in range(0,len(y_predict_gbc)):
    a=ytest[i]
    b=y_predict_gbc[i]
    L[a][b]+=1


# In[25]:


plt.bar(x, L[0], color="blue")
plt.title("Prediction of 0")
plt.xlim([0,9])
plt.show()


# In[26]:


plt.bar(x, L[1], color="blue")
plt.title("Prediction of 1")
plt.xlim([0,9])
plt.show()


# In[27]:


plt.bar(x, L[2], color="blue")
plt.title("Prediction of 2")
plt.xlim([0,9])
plt.show()


# In[28]:


plt.bar(x, L[3], color="blue")
plt.title("Prediction of 3")
plt.xlim([0,9])
plt.show()


# In[29]:


plt.bar(x, L[4], color="blue")
plt.title("Prediction of 4")
plt.xlim([0,9])
plt.show()


# In[30]:


plt.bar(x, L[5], color="blue")
plt.title("Prediction of 5")
plt.xlim([0,9])
plt.show()


# In[31]:


plt.bar(x, L[6], color="blue")
plt.title("Prediction of 6")
plt.xlim([0,9])
plt.show()


# In[32]:


plt.bar(x, L[7], color="blue")
plt.title("Prediction of 7")
plt.xlim([0,9])
plt.show()


# In[33]:


plt.bar(x, L[8], color="blue")
plt.title("Prediction of 8")
plt.xlim([0,9])
plt.show()


# In[34]:


plt.bar(x, L[9], color="blue")
plt.title("Prediction of 9")
plt.xlim([0,9])
plt.show()