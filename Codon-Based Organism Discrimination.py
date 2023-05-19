#%%
##############################################################################
#=====================================Q1=====================================#
##############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Importing the data
data = pd.read_csv("G:/UH/S2/MATH6373-Deep Learning and Artificial Neural Networks/HW/codon_usage.csv/codon_usage.csv")

#General Stats
#data.shape (13028, 69)

#len(data["SpeciesID"].unique()) 12368

#len(data["SpeciesName"].unique()) 13016

#len(data["Kingdom"].unique()) 11

#len(data["DNAtype"].unique() 11

codon = data.drop(["SpeciesID","SpeciesName"], axis=1)

#One Hot Encoding Categorical Variable DNAType
new = codon
new["DNAtype0-genomic"] = 0
new["DNAtype1-mitochondrial"] = 0
new["DNAtype2-chloroplast"] = 0
new["DNAtype3-cyanelle"] = 0
new["DNAtype4-plastid"] = 0
new["DNAtype5-nucleomorph"] = 0
new["DNAtype6-secondary_endosymbiont"] = 0
new["DNAtype7-chromoplast"] = 0
new["DNAtype8-leucoplast"] = 0
new["DNAtype9-NA"] = 0
new["DNAtype10-proplastid"] = 0
new["DNAtype11-apicoplast"] = 0
new["DNAtype12-kinetoplast"] = 0

new.loc[new["DNAtype"] == 0,["DNAtype0-genomic"]] = 1
new.loc[new["DNAtype"] == 1,["DNAtype1-mitochondrial"]] = 1
new.loc[new["DNAtype"] == 2,["DNAtype2-chloroplast"]] = 1
new.loc[new["DNAtype"] == 3,["DNAtype3-cyanelle"]] = 1
new.loc[new["DNAtype"] == 4,["DNAtype4-plastid"]] = 1
new.loc[new["DNAtype"] == 5,["DNAtype5-nucleomorph"]] = 1
new.loc[new["DNAtype"] == 6,["DNAtype6-secondary_endosymbiont"]] = 1
new.loc[new["DNAtype"] == 7,["DNAtype7-chromoplast"]] = 1
new.loc[new["DNAtype"] == 8,["DNAtype8-leucoplast"]] = 1
new.loc[new["DNAtype"] == 9,["DNAtype9-NA"]] = 1
new.loc[new["DNAtype"] == 10,["DNAtype10-proplastid"]] = 1
new.loc[new["DNAtype"] == 11,["DNAtype11-apicoplast"]] = 1
new.loc[new["DNAtype"] == 12,["DNAtype12-kinetoplast"]] = 1

new = new.drop(["DNAtype"], axis=1)

#Scaling Ncodons
scaler = MinMaxScaler()
new["Ncodons"]  = scaler.fit_transform(new["Ncodons"] .values.reshape(-1,1))

#Imbalanced
new["Kingdom"].value_counts()

Y = new["Kingdom"]
X = new.drop(["Kingdom"], axis=1)
X.apply(pd.to_numeric, errors="ignore")
X = X.replace("non-B hepatitis virus", 0)
X = X.replace("12;I", 0)
X = X.replace("-", 0)
X.astype(float)
X["UUU"] = pd.to_numeric(X["UUU"])
X["UUC"] = pd.to_numeric(X["UUC"])

sm = SMOTE(random_state = 2)
X,Y = sm.fit_resample(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()




#%%
##############################################################################
#=====================================Q2=====================================#
##############################################################################
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown="ignore")
Y_train = encoder.fit_transform(Y_train.to_numpy().reshape(-1,1)).toarray()
Y_test = encoder.fit_transform(Y_test.to_numpy().reshape(-1,1)).toarray()

#Defining the number of input features (or Input Dimensions)
p = X_train.shape[1] #p = 78

'''
#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#

mlp_star = tf.keras.Sequential()
mlp_star.add(tf.keras.layers.Dense(units = p/2, activation= "relu", input_dim = p))
mlp_star.add(tf.keras.layers.Dense(units = 11, activation= "softmax"))

mlp_star.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")
history_star = mlp_star.fit(X_train, Y_train, epochs=2000,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)


#Plot 1
pd.DataFrame(history_star.history)[["loss","val_loss"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Plot 2
pd.DataFrame(history_star.history)[["categorical_accuracy","val_categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Accuracy Increase per 500 epochs
history_star.history['val_categorical_accuracy'][::200]
#Good number of epochs seems to be 1400


#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#
'''
#%%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import multilabel_confusion_matrix


#Trained Model Against the Test Set
mlp_star = tf.keras.Sequential()
mlp_star.add(tf.keras.layers.Dense(units = p/2, activation= "relu", input_dim = p))
mlp_star.add(tf.keras.layers.Dense(units = 11, activation= "softmax"))

mlp_star.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")
history_star = mlp_star.fit(X_train, Y_train, epochs=1400,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)
history_star.history


#Confusion Matrix
pred_star = mlp_star.predict(X_test)
pred_star

pred_star[0]

tf.math.argmax(pred_star,axis=1)

tf.one_hot(
    tf.math.argmax(pred_star,axis=1),
    depth=11
)

#Combine output to onehot
pred_star_one_hot = tf.one_hot(tf.math.argmax(pred_star,axis=1), depth=Y_test.shape[1])
pred_star_one_hot = pred_star_one_hot.numpy()


multilabel_confusion_matrix(Y_test, pred_star_one_hot)

#Fancy Code for Confusion Matrices
def plot_confusion_matrix(Y_test, pred_star_one_hot, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, pred_star_one_hot)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(Y_test, pred_star_one_hot)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(Y))-0.5)
    plt.ylim(len(np.unique(Y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)
class_names = data["Kingdom"].unique()

#Normalized Confusion Matrix
plot_confusion_matrix(Y_test.argmax(axis = 1), pred_star_one_hot.argmax(axis = 1), classes=class_names, normalize=True,
                      title="Normalized Confusion Matrix (MLP*)")


#%%
##############################################################################
#=====================================Q3=====================================#
##############################################################################

from sklearn.decomposition import PCA

#Recalling and modifying X from Q1
new_X = pd.DataFrame(X[:])


#PCA Analysis
pca = PCA()
pca.fit(new_X)
pca_data = pca.transform(new_X)

pev = np.round(pca.explained_variance_ratio_, decimals = 3)
pev
pev = pev.reshape(13,6)
#4 Principal Components

#%%
'''
#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#

mlp_low = tf.keras.Sequential()
mlp_low.add(tf.keras.layers.Dense(units = 4, activation= "relu", input_dim = p))
mlp_low.add(tf.keras.layers.Dense(units = 11, activation= "softmax"))

mlp_low.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")

history_low = mlp_low.fit(X_train, Y_train, epochs=2000,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)

#Plot 1
pd.DataFrame(history_low.history)[["loss","val_loss"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Plot 2
pd.DataFrame(history_low.history)[["categorical_accuracy","val_categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Accuracy Increase per 500 epochs
history_low.history['val_categorical_accuracy'][::200]
#5000 is again a safe choice

#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#
'''

#%%

#Trained Model Against the Test Set
mlp_low = tf.keras.Sequential()
mlp_low.add(tf.keras.layers.Dense(units = 4, activation= "relu", input_dim = p))
mlp_low.add(tf.keras.layers.Dense(units = 11, activation= "softmax"))

mlp_low.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")

history_low = mlp_low.fit(X_train, Y_train, epochs=1800,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)
history_low.history


#Plot 1
pd.DataFrame(history_low.history)[["loss","val_loss"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Plot 2
pd.DataFrame(history_low.history)[["categorical_accuracy","val_categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()




#Confusion Matrix
pred_low = mlp_low.predict(X_test)
pred_low

pred_low[0]

tf.math.argmax(pred_low,axis=1)


#Combine output to onehot
pred_low_one_hot = tf.one_hot(tf.math.argmax(pred_low,axis=1), depth=Y_test.shape[1])
pred_low_one_hot = pred_low_one_hot.numpy()


multilabel_confusion_matrix(Y_test, pred_low_one_hot)

#Fancy Code for Confusion Matrices
def plot_confusion_matrix2(Y_test, pred_low_one_hot, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, pred_low_one_hot)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(Y_test, pred_low_one_hot)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(Y))-0.5)
    plt.ylim(len(np.unique(Y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)
class_names = data["Kingdom"].unique()

#Normalized Confusion Matrix
plot_confusion_matrix2(Y_test.argmax(axis = 1), pred_low_one_hot.argmax(axis = 1), classes=class_names, normalize=True,
                      title="Normalized Confusion Matrix (MLP_low)")
#The results with the principal components in place of all the features appears
#to have performed worse.

#%%
##############################################################################
#=====================================Q4=====================================#
##############################################################################

#Recalling and modifying new

XY = pd.DataFrame(Y[:])
XY = pd.concat([XY,X[:]], axis = 1)

X_cl1 = XY.loc[XY['Kingdom'] == "vrl"]
X_cl1 = X_cl1.loc[:, (X_cl1 != 0).any(axis=0)]
X_cl1 = X_cl1.drop(["Kingdom"], axis = 1)

X_cl2 = XY.loc[XY['Kingdom'] == "arc"]
X_cl2 = X_cl2.loc[:, (X_cl2 != 0).any(axis=0)]
X_cl2 = X_cl2.drop(["Kingdom"], axis = 1)

X_cl3 = XY.loc[XY['Kingdom'] == "bct"]
X_cl3 = X_cl3.loc[:, (X_cl3 != 0).any(axis=0)]
X_cl3 = X_cl3.drop(["Kingdom"], axis = 1)

X_cl4 = XY.loc[XY['Kingdom'] == "phg"]
X_cl4 = X_cl4.loc[:, (X_cl4 != 0).any(axis=0)]
X_cl4 = X_cl4.drop(["Kingdom"], axis = 1)

X_cl5 = XY.loc[XY['Kingdom'] == "plm"]
X_cl5 = X_cl5.loc[:, (X_cl5 != 0).any(axis=0)]
X_cl5 = X_cl5.drop(["Kingdom"], axis = 1)

X_cl6 = XY.loc[XY['Kingdom'] == "pln"]
X_cl6 = X_cl6.loc[:, (X_cl6 != 0).any(axis=0)]
X_cl6 = X_cl6.drop(["Kingdom"], axis = 1)

X_cl7 = XY.loc[XY['Kingdom'] == "inv"]
X_cl7 = X_cl7.loc[:, (X_cl7 != 0).any(axis=0)]
X_cl7 = X_cl7.drop(["Kingdom"], axis = 1)

X_cl8 = XY.loc[XY['Kingdom'] == "vrt"]
X_cl8 = X_cl8.loc[:, (X_cl8 != 0).any(axis=0)]
X_cl8 = X_cl8.drop(["Kingdom"], axis = 1)

X_cl9 = XY.loc[XY['Kingdom'] == "mam"]
X_cl9 = X_cl9.loc[:, (X_cl9 != 0).any(axis=0)]
X_cl9 = X_cl9.drop(["Kingdom"], axis = 1)

X_cl10 = XY.loc[XY['Kingdom'] == "rod"]
X_cl10 = X_cl10.loc[:, (X_cl10 != 0).any(axis=0)]
X_cl10 = X_cl10.drop(["Kingdom"], axis = 1)

X_cl11 = XY.loc[XY['Kingdom'] == "pri"]
X_cl11 = X_cl11.loc[:, (X_cl11 != 0).any(axis=0)]
X_cl11 = X_cl11.drop(["Kingdom"], axis = 1)


#PCA Analysis per Class
pca.fit(X_cl1)
pca_data = pca.transform(X_cl1)
pev_cl1 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl1[:28])


pca.fit(X_cl2)
pca_data = pca.transform(X_cl2)
pev_cl2 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl2[:7])


pca.fit(X_cl3)
pca_data = pca.transform(X_cl3)
pev_cl3 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl3[:14])


pca.fit(X_cl4)
pca_data = pca.transform(X_cl4)
pev_cl4 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl4[:10])


pca.fit(X_cl5)
pca_data = pca.transform(X_cl5)
pev_cl5 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl5[:5])


pca.fit(X_cl6)
pca_data = pca.transform(X_cl6)
pev_cl6 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl6[:2])


pca.fit(X_cl7)
pca_data = pca.transform(X_cl7)
pev_cl7 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl7[:1])


pca.fit(X_cl8)
pca_data = pca.transform(X_cl8)
pev_cl8 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl8[:1])


pca.fit(X_cl9)
pca_data = pca.transform(X_cl9)
pev_cl9 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl9[:1])


pca.fit(X_cl10)
pca_data = pca.transform(X_cl10)
pev_cl10 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl10[:1])


pca.fit(X_cl11)
pca_data = pca.transform(X_cl11)
pev_cl11 = np.round(pca.explained_variance_ratio_, decimals = 3)
sum(pev_cl1[:28])

h_high = 28 + 7 + 14 + 10 + 5 + 2 + 1 + 1 + 1 + 1 + 28
h_high


#%%
'''
#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#

mlp_high = tf.keras.Sequential()
mlp_high.add(tf.keras.layers.Dense(units = 98, activation= "relu", input_dim = p))
mlp_high.add(tf.keras.layers.Dense(units = 11, activation= "softmax"))

mlp_high.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")

history_high = mlp_high.fit(X_train, Y_train, epochs=2000,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)

#Plot 1
pd.DataFrame(history_high.history)[["loss","val_loss"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Plot 2
pd.DataFrame(history_high.history)[["categorical_accuracy","val_categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Accuracy Increase per 500 epochs
history_high.history['val_categorical_accuracy'][::200]
#1600 is a good choice this time
#Testing (REMOVE """ FOR RESULTS) Testing (REMOVE """ FOR RESULTS) Testing#
'''
#%%
#Trained Model Against the Test Set
mlp_high = tf.keras.Sequential()
mlp_high.add(tf.keras.layers.Dense(units = 98, activation= "relu", input_dim = p, name = "H1"))
mlp_high.add(tf.keras.layers.Dense(units = 11, activation= "softmax", name = "Out"))

mlp_high.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")

history_high = mlp_high.fit(X_train, Y_train, epochs=1600,batch_size= 1042,
    validation_data = (X_test, Y_test), verbose=1)
history_high.history


#Accuracy Increase per 500 epochs
history_high.history['val_categorical_accuracy'][::200]

#Confusion Matrix
pred_high = mlp_high.predict(X_test)
pred_high

pred_high[0]

tf.math.argmax(pred_high,axis=1)


#Combine output to onehot
pred_high_one_hot = tf.one_hot(tf.math.argmax(pred_high,axis=1), depth=Y_test.shape[1])
pred_high_one_hot = pred_high_one_hot.numpy()


multilabel_confusion_matrix(Y_test, pred_high_one_hot)

#Fancy Code for Confusion Matrices
def plot_confusion_matrix2(Y_test, pred_high_one_hot, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, pred_high_one_hot)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(Y_test, pred_high_one_hot)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(Y))-0.5)
    plt.ylim(len(np.unique(Y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)
class_names = data["Kingdom"].unique()

#Normalized Confusion Matrix
plot_confusion_matrix2(Y_test.argmax(axis = 1), pred_high_one_hot.argmax(axis = 1), classes=class_names, normalize=True,
                      title="Normalized Confusion Matrix (mlp_high)")

#MLP_high is the best performer

#%%
##############################################################################
#=====================================Q5=====================================#
##############################################################################


Zn_train = mlp_high.layers[0](X_train)
Zn_test = mlp_high.layers[0](X_test)
X = X.to_numpy()
Zn = mlp_high.layers[0](X)

#Explain in the report what is the principle of sparsity learning.
#the features of the data should be kept with droping useless neurons



sparsity_regularizer = tf.keras.regularizers.l1(0.1)
aec = tf.keras.Sequential([
    tf.keras.layers.Dense(98, activation='relu', activity_regularizer=sparsity_regularizer ,name='HK'),
    tf.keras.layers.Dense(98, activation='softmax', name='KH')
])
aec.compile(optimizer='adam', loss='mse', metrics = "categorical_accuracy")
AEC = aec.fit(Zn_train, Zn_train, epochs=100,batch_size= 104)
#Plot 1
pd.DataFrame(AEC.history)["categorical_accuracy"].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()


sparsity_regularizer = tf.keras.regularizers.l1(0.1)
aec_train = tf.keras.Sequential([
    tf.keras.layers.Dense(98, activation='relu', activity_regularizer=sparsity_regularizer ,name='HK'),
    tf.keras.layers.Dense(98, activation='softmax', name='KH')
])
aec_train.compile(optimizer='adam', loss='mse', metrics = "categorical_accuracy")
AEC_train = aec_train.fit(Zn_train, Zn_train, epochs=100,batch_size= 104)



aec_test = tf.keras.Sequential([
    tf.keras.layers.Dense(98, activation='relu', activity_regularizer=sparsity_regularizer ,name='HK'),
    tf.keras.layers.Dense(98, activation='softmax', name='KH')
])
aec_test.compile(optimizer='adam', loss='mse', metrics = "categorical_accuracy")
AEC_test = aec_test.fit(Zn_test, Zn_test, epochs=100,batch_size= 104)

#Plot RMSE_train and RMSE_test
plt.plot( pd.DataFrame(AEC_train.history)[["categorical_accuracy"]], color='red', lw=3,label = "Train")
plt.plot( pd.DataFrame(AEC_test.history)[["categorical_accuracy"]], color='orange', lw=3,label = "Test")
plt.xlabel("epoch")
plt.legend()
plt.show()



#mAEC  = 30



from numpy import linalg as LA


M = LA.norm(Zn)

RAEC = pd.DataFrame(AEC_test.history)["loss"][40]/M
RAEC#1.9590243245740492e-05

#%%
##############################################################################
#=====================================Q6=====================================#
##############################################################################

#Combining MLP_high and AEC*

H1 = mlp_high.get_layer("H1")
HK = aec_test.get_layer("HK")
KH= aec_test.get_layer("KH")

MLP12 = tf.keras.Sequential()
MLP12.add(H1)
MLP12.add(HK)
MLP12.add(tf.keras.layers.Dense(units = 11, activation= "softmax", name = "Out"))


MLP12.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy", run_eagerly= True)


#Testing
history_MLP12 = MLP12.fit(X_train, Y_train, epochs=500, batch_size = 1042, validation_data = (X_test, Y_test), verbose=1)

pd.DataFrame(history_MLP12.history)[["categorical_accuracy", "val_categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

#Accuracy Increase per 500 epochs
history_MLP12.history['val_categorical_accuracy'][::50]
#3000 Epochs seems enough for the test set


#%%
#Creating MLP3
##############################################################################
Un = MLP12.layers[0](X)
Un = Un.numpy()


m2 = tf.keras.Model(inputs=MLP12.input, outputs = MLP12.get_layer('Out').output)
Un2 = m2.predict(X)

Un.shape
Un2.shape

MLP3 = tf.keras.Sequential()
MLP3.add(MLP12.layers[0])
MLP3.add(tf.keras.layers.Dense(units = 52, activation= "relu", name = "G"))
MLP3.add(tf.keras.layers.Dense(11, activation='softmax', name='Out'))


Y_n = encoder.fit_transform(Y.to_numpy().reshape(-1,1)).toarray()
MLP3.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")
history_MLP3 = MLP3.fit(X_train, Y_train, epochs=500,batch_size= 1042)

pd.DataFrame(history_MLP3.history)[["categorical_accuracy"]].plot()
plt.xlabel("epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()
#50 epochs looks fine


#Confusion Matrix
pred_MLP3 = MLP3.predict(X_test)
pred_MLP3

pred_MLP3[0]

tf.math.argmax(pred_MLP3,axis=1)

tf.one_hot(
    tf.math.argmax(pred_MLP3,axis=1),
    depth=11
)

#Combine output to onehot
pred_MLP3_one_hot = tf.one_hot(tf.math.argmax(pred_MLP3,axis=1), depth=Y_test.shape[1])
pred_MLP3_one_hot = pred_MLP3_one_hot.numpy()


multilabel_confusion_matrix(Y_test, pred_MLP3_one_hot)

#Fancy Code for Confusion Matrices
def plot_confusion_matrix(Y_test, pred_MLP3_one_hot, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, pred_MLP3_one_hot)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(Y_test, pred_MLP3_one_hot)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(Y))-0.5)
    plt.ylim(len(np.unique(Y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)
class_names = data["Kingdom"].unique()

#Normalized Confusion Matrix
plot_confusion_matrix(Y_test.argmax(axis = 1), pred_MLP3_one_hot.argmax(axis = 1), classes=class_names, normalize=True,
                      title="Normalized Confusion Matrix (MLP3*)")


#%%

G = MLP3.get_layer("G")
OUT = MLP3.get_layer("Out")

MLP_long = tf.keras.Sequential()
MLP_long.add(H1)
MLP_long.add(HK)
MLP_long.add(G)
MLP_long.add(OUT)

MLP_long.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "categorical_accuracy")
history_MLP_long = MLP3.fit(X_train, Y_train, epochs=500,batch_size= 1042,
                            validation_data = (X_test, Y_test), verbose=1)



#Confusion Matrix
pred_MLP_long = MLP_long.predict(X_test)
pred_MLP_long

pred_MLP_long[0]

tf.math.argmax(pred_MLP_long,axis=1)


#Combine output to onehot
pred_MLP_long_one_hot = tf.one_hot(tf.math.argmax(pred_MLP_long,axis=1), depth=Y_test.shape[1])
pred_MLP_long_one_hot = pred_MLP_long_one_hot.numpy()


multilabel_confusion_matrix(Y_test, pred_MLP_long_one_hot)

#Fancy Code for Confusion Matrices
def plot_confusion_matrix2(Y_test, pred_MLP_long_one_hot, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, pred_MLP_long_one_hot)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(Y_test, pred_MLP_long_one_hot)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(Y))-0.5)
    plt.ylim(len(np.unique(Y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)
class_names = data["Kingdom"].unique()

#Normalized Confusion Matrix
plot_confusion_matrix2(Y_test.argmax(axis = 1), pred_MLP_long_one_hot.argmax(axis = 1), classes=class_names, normalize=True,
                      title="Normalized Confusion Matrix (MLP_long)")

#MLP_MLP_long is the best performer