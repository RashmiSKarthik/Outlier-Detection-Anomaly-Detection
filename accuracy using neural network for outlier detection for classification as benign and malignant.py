import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
np.random.seed(697)
data = pd.read_csv('F:/Implementation/lung_stats.csv', header = 0)
plt.scatter(data['lung_mean_hu'], data['lung_pd05_hu'])
plt.xlabel('lung_mean_hu')
plt.ylabel('lung_pd05_hu')
plt.show()

data = data.drop(['img_id'], axis = 1)
x=data.copy()
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(x)
#kmeans(algorithm='auto',copy_x=True, init='K-means++', max_iter=300,
  #     n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto')
clusters=x.copy()
clusters['cluster_pred']= kmeans.fit_predict(x)
plt.scatter(clusters['lung_mean_hu'], clusters['lung_pd05_hu'], 
            c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel=['lung_mean_hu']
plt.ylabel=['lung_pd05_hu']
plt.show()


print(data.lung_pd95_hu.value_counts())




data = data.drop(['img_id'], axis = 1)

features = ['lung_area_mm2', 'lung_volume_fraction', 'lung_mean_hu', 
        'lung_pd05_hu']

data = pd.get_dummies(data, columns = features)     
#Split in 75% train and 25% test set
X, y = train_test_split(data, test_size = 0.25, random_state= 1984)

#Make sure labels are equally distributed in train and test set
X.lung_pd95_hu.sum()/X.shape[0]
y.lung_pd95_hu.sum()/y.shape[0] 

#Get the data ready for the Neural Network
training_y = X.lung_pd95_hu
testing_y = y.lung_pd95_hu

training_x =X.drop(['lung_pd95_hu'], axis = 1)
testing_x = y.drop(['lung_pd95_hu'], axis = 1)

training_x =np.array(training_x)
testing_x = np.array(testing_x)

training_y = np.array(training_y)
testing_y = np.array(testing_y)
#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
adam = optimizers.adam(lr = 0.005, decay = 0.0000001)

model = Sequential()
model.add(Dense(48, input_dim=training_x.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(24,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="tanh"))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(training_x, training_y, validation_split=0.2, epochs=3, batch_size=64)


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict on test set
predictions_NN_prob = model.predict(testing_x)
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#Print accuracy
#acc_NN = accuracy_score(testing_y, predictions_NN_01)
#print('Overall accuracy of Neural Network model:', acc_NN)

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(testing_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.grid(b=None)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('TRUE POSITIVE RATE')
plt.xlabel('FALSE POSITIVE RATE')
plt.show()

#Print Confusion Matrix
cm = confusion_matrix(testing_y, predictions_NN_01)
labels = ['BENIGN', 'MALIGNANT']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('CONFUSION MATRIX')
plt.ylabel('TRUE CLASS')
plt.xlabel('PREDICTED CLASS')
plt.show()
