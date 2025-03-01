import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
  #  for filename in filenames:
   #     print(os.path.join(dirname, filename))
   
#pip install seaborn

imsize = 448
imchanel = 3
labels = ['daisy','dandelion','rose','sunflower','tulip']

train_dir = '/kaggle/input/flowers-recognition/flowers/'
test_dir = '/kaggle/input/flowers-recognition/flowers/'
valid_dir = '/kaggle/input/flowers-recognition/flowers/'

# dont worry. we will use train_test split in train data for seperate validation and test sets


import os
from IPython.display import display
import warnings
import cv2
import time
import shutil
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
sns.set_style('darkgrid')
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import regularizers
from keras.optimizers import Adam, Adamax
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_crossentropy
from keras.models import Model, load_model, Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
def define_paths(dir):
    filepaths = []
    labels = []
    folds = os.listdir(dir)
    for fold in folds:
        foldpath = os.path.join(dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return filepaths, labels

def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

def create_df(tr_dir, val_dir, ts_dir):
    # train dataframe
    files, classes = define_paths(tr_dir)
    train_df = define_df(files, classes)

    # validation dataframe
    files, classes = define_paths(val_dir)
    valid_df = define_df(files, classes)
    # test dataframe
    files, classes = define_paths(ts_dir)
    test_df = define_df(files, classes)
    return train_df, valid_df, test_df

def create_gens(train_df, valid_df, test_df, batch_size):
    img_size = (imsize, imsize)
    channels = imchanel
    img_shape = (img_size[0], img_size[1], channels)
    ts_length = len(test_df)
    test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size
    def scalar(img):
        return img

        
    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True ,vertical_flip=True, rotation_range=1,)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= False ,rotation_range=0.4)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= test_batch_size)
    return train_gen, valid_gen, test_gen


def show_images(gen):
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes)
    images, labels = next(gen)        # get a batch size samples from the generator
    plt.figure(figsize= (20, 20))
    length = len(labels)              # length of batch size
    sample = min(length, 25)          # check if sample less than 25 images
    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()

def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)     # get number of epoch with the lowest validation loss
    val_lowest = val_loss[index_loss]    # get the loss value of epoch with the lowest validation loss
    index_acc = np.argmax(val_acc)       # get number of epoch with the highest validation accuracy
    acc_highest = val_acc[index_acc]     # get the loss value of epoch with the highest validation accuracy

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]	       # create x-axis by epochs count
    loss_label = f'best epoch= {str(index_loss + 1)}'  # label of lowest val_loss
    acc_label = f'best epoch= {str(index_acc + 1)}'    # label of highest val_accuracy
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout
    plt.show()

def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix', cmap= plt.cm.Blues):
    plt.figure(figsize= (10, 10))
    plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, Without Normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.grid(None)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, base_model, patience, stop_patience, threshold, factor, dwell, batches, initial_epoch, epochs, ask_epoch):
        super(MyCallback, self).__init__()
        self.model = model
        self.base_model = base_model
        self.patience = patience # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience # specifies how many times to adjust lr without improvement to stop training
        self.threshold = threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = factor # factor by which to reduce the learning rate
        self.dwell = dwell
        self.batches = batches # number of training batch to runn per epoch
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch # save this value to restore if restarting training
        # callback variables
        self.count = 0 # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1   # epoch with the lowest loss
        self.initial_lr = float(tf.keras.backend.get_value(model.optimizer.lr)) # get the initial learning rate and save it
        self.highest_tracc = 0.0 # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf # set lowest validation loss to infinity initially
        self.best_weights = self.model.get_weights() # set best weights to model's initial weights
        self.initial_weights = self.model.get_weights()   # save initial weights if they have to get restored

    # Define a function that will run when train begins
    def on_train_begin(self, logs= None):
        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor','% Improv', 'Duration')
        print(msg)
        self.start_time = time.time()

    def on_train_end(self, logs= None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg)
        self.model.set_weights(self.best_weights) # set the weights of the model to the best weights

    def on_train_batch_end(self, batch, logs= None):
        acc = logs.get('accuracy') * 100 # get batch accuracy
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:} of {2:5s}-   accuracy=  {3:5.3f}   -   loss: {4:8.5f}'.format(' ', str(batch), str(self.batches), acc, loss)
        print(msg, '\r', end= '') # prints over on the same line to show running batch count

    def on_epoch_begin(self, epoch, logs= None):
        self.ep_start = time.time()

    # Define method runs on the end of each epoch
    def on_epoch_end(self, epoch, logs= None):
        ep_end = time.time()
        duration = ep_end - self.ep_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
        current_lr = lr
        acc = logs.get('accuracy')  # get training accuracy
        v_acc = logs.get('val_accuracy')  # get validation accuracy
        loss = logs.get('loss')  # get training loss for this epoch
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch

        if acc < self.threshold: # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (acc - self.highest_tracc ) * 100 / self.highest_tracc # define improvement of model progres

            if acc > self.highest_tracc: # training accuracy improved in the epoch
                self.highest_tracc = acc # set new highest training accuracy
                self.best_weights = self.model.get_weights() # training accuracy improved so save the weights
                self.count = 0 # set count to 0 since training accuracy improved
                self.stop_count = 0 # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = epoch + 1  # set the value of best epoch for this epoch

            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1: # lr should be adjusted
                    lr = lr * self.factor # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer
                    self.count = 0 # reset the count to 0
                    self.stop_count = self.stop_count + 1 # count the number of consecutive lr adjustments
                    self.count = 0 # reset counter
                    if self.dwell:
                        self.model.set_weights(self.best_weights) # return to better point in N space
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1 # increment patience counter

        else: # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (self.lowest_vloss - v_loss ) * 100 / self.lowest_vloss
            if v_loss < self.lowest_vloss: # check if the validation loss improved
                self.lowest_vloss = v_loss # replace lowest validation loss with new validation loss
                self.best_weights = self.model.get_weights() # validation loss improved so save the weights
                self.count = 0 # reset count since validation loss improved
                self.stop_count = 0
                self.best_epoch = epoch + 1 # set the value of the best epoch to this epoch
            else: # validation loss did not improve
                if self.count >= self.patience - 1: # need to adjust lr
                    lr = lr * self.factor # adjust the learning rate
                    self.stop_count = self.stop_count + 1 # increment stop counter because lr was adjusted
                    self.count = 0 # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(self.best_weights) # return to better point in N space
                else:
                    self.count = self.count + 1 # increment the patience counter
                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        print(msg)

        if self.stop_count > self.stop_patience - 1: # check if learning rate has been adjusted stop_count times with no improvement
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print(msg)
            self.model.stop_training = True # stop training

        else:
            if self.ask_epoch != None:
                if epoch + 1 >= self.ask_epoch:
                    msg = 'enter H to halt training or an integer for number of epochs to run then ask again'
                    print(msg)
                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'training has been halted at epoch {epoch + 1} due to user input'
                        print(msg)
                        self.model.stop_training = True # stop training
                    else:
                        try:
                            ans = int(ans)
                            self.ask_epoch += ans
                            msg = f' training will continue until epoch ' + str(self.ask_epoch)
                            print(msg)
                            msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
                            print(msg)
                        except:
                            print('Invalid')
                            
from tensorflow.keras.preprocessing.image import ImageDataGenerator
DATA_DIR = R"C:\Users\rms11\Desktop\Proj\Datasets\flowers"
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)
train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123)
valid_df, test_df = train_test_split(dummy_df,  train_size= 0.6, shuffle= True, random_state= 123)
batch_size = 40

# Calculate frequency of each label in train_df
grouped_train = train_df.groupby('labels').size()

# Generate a gradient of colors for the bars
colors = np.linspace(0, 1, len(grouped_train))

# Plot bar chart of label frequencies with gradient colors
plt.bar(grouped_train.index, grouped_train.values, color=plt.cm.cool(colors))
plt.title('Frequency of labels in train_df')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()

grouped_train = train_df.groupby('labels').size()
grouped_valid = valid_df.groupby('labels').size()
grouped_test = test_df.groupby('labels').size()
max_count = max(max(grouped_train), max(grouped_valid), max(grouped_test))


labels = grouped_train.index.tolist()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

colors_train = np.linspace(0, 1, len(grouped_train))
ax1.bar(grouped_train.index, grouped_train.values, color=plt.cm.cool(colors_train))
ax1.set_title('Train Data')
ax1.set_xlabel('Labels')
ax1.set_ylabel('Frequency')
ax1.set_xticklabels(labels, rotation=45)

colors_valid = np.linspace(0, 1, len(grouped_valid))
ax2.bar(grouped_valid.index, grouped_valid.values, color=plt.cm.cool(colors_valid))
ax2.set_title('Validation Data')
ax2.set_xlabel('Labels')
ax2.set_ylabel('Frequency')
ax2.set_xticklabels(labels, rotation=45)


colors_test = np.linspace(0, 1, len(grouped_test))
ax3.bar(grouped_test.index, grouped_test.values, color=plt.cm.cool(colors_test))
ax3.set_title('Test Data')
ax3.set_xlabel('Labels')
ax3.set_ylabel('Frequency')
ax3.set_xticklabels(labels, rotation=45)


plt.tight_layout()


plt.show()


# Calculate the frequency of labels in each dataset
grouped_train = train_df['labels'].value_counts()
grouped_valid = valid_df['labels'].value_counts()
grouped_test = test_df['labels'].value_counts()

# Determine the maximum count of all datasets
max_count = max(grouped_train.max(), grouped_valid.max(), grouped_test.max())

# Get the labels
labels = grouped_train.index.tolist()

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(grouped_train.index, grouped_train.values, color='tab:blue', alpha=0.7, label='Train')
ax.bar(grouped_valid.index, grouped_valid.values, color='tab:orange', alpha=0.7, label='Validation')
ax.bar(grouped_test.index, grouped_test.values, color='tab:green', alpha=0.7, label='Test')

# Set the axis labels and title
ax.set_title('Frequency of Labels in All Datasets', fontsize=14, fontweight='bold')
ax.set_xlabel('Labels', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# Set the ticks and limits
ax.set_xticklabels(labels, fontsize=10, rotation=45)
ax.set_ylim([0, max_count*1.1])
ax.tick_params(axis='both', which='major', labelsize=10)

# Add a legend
ax.legend(loc='upper right')

plt.show()

# Calculate the frequency of labels in each dataset
grouped_train = train_df['labels'].value_counts()
grouped_valid = valid_df['labels'].value_counts()
grouped_test = test_df['labels'].value_counts()

# Determine the maximum count of all datasets
max_count = max(grouped_train.max(), grouped_valid.max(), grouped_test.max())

# Get the labels
labels = grouped_train.index.tolist()

# Create a pie plot
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
ax.pie([grouped_train.sum(), grouped_valid.sum(), grouped_test.sum()], labels=['Train', 'Validation', 'Test'], autopct='%1.1f%%', startangle=90, colors=colors)

# Set the title
ax.set_title('Frequency of Labels in All Dataset', fontsize=14, fontweight='bold')

plt.show()


# Calculate the frequency of each label in each dataset
grouped_train = train_df['labels'].value_counts()
grouped_valid = valid_df['labels'].value_counts()
grouped_test = test_df['labels'].value_counts()

# Get the labels
labels = list(set(train_df['labels'].unique()) | set(valid_df['labels'].unique()) | set(test_df['labels'].unique()))

# Create a pie plot for each label
fig, axs = plt.subplots(1, len(labels), figsize=(15, 5))

for ax, label in zip(axs, labels):
    # Get the frequency of the current label in each dataset
    freqs = [
        grouped_train.get(label, 0),
        grouped_valid.get(label, 0),
        grouped_test.get(label, 0)
    ]
    
    # Create a pie plot
    ax.pie(
        freqs,
        labels=['Train', 'Validation', 'Test'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['tab:blue', 'tab:orange', 'tab:green']
    )
    
    # Set the title
    ax.set_title(label, fontsize=12, fontweight='bold')
    
# Set the suptitle
fig.suptitle('Frequency of Labels in All Dataset by Label', fontsize=16, fontweight='bold')

plt.show()

from tensorflow.keras import layers
from keras import optimizers
from tensorflow.keras import initializers
batch_size = 40
train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)
img_size = (imsize, imsize)
channels = imchanel
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys()))
base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top= False, input_shape= img_shape, pooling= 'max')

#block7a_expand_conv
for layer in base_model.layers:
  if layer.name == 'block7d_project_conv':
    break
  layer.trainable = False
  #print('Layer ' + layer.name + ' frozen.')     




x = layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)(base_model.output)
x = layers.Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.1),
                     bias_initializer=initializers.Zeros())(x)
x = layers.Dropout(rate= 0.45, seed= 123)(x)                
x = layers.Dense(class_count, activation= 'softmax')(x)  

model_incep = tf.keras.models.Model(base_model.input, x)

model_incep.compile(optimizers.Adamax(learning_rate= 0.0011), 
                    loss = 'mae', 
                    metrics = ['accuracy'])

batch_size = 40
epochs = 20
patience = 1 		# number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3 	# number of epochs to wait before stopping training if monitored value does not improve
threshold = 0.9 	# if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor = 0.5 		# factor to reduce lr by
dwell = True 		# experimental, if True and monitored metric does not improve on current epoch set  modelweights back to weights of previous epoch
freeze = False 		# if true free weights of  the base model
ask_epoch = None		# number of epochs to run before asking if you want to halt training
batches = int(np.ceil(len(train_gen.labels) / batch_size))


checkpoint_callback2 = keras.callbacks.ModelCheckpoint(
              filepath="gdrive/My Drive/...",
              monitor= 'val_accuracy', 
              verbose= 1,
              save_best_only= True, 
              mode = 'auto'
              );

callbacks = [MyCallback(model= model_incep, base_model= base_model, patience= patience,
            stop_patience= stop_patience, threshold= threshold, factor= factor,
            dwell= dwell, batches= batches, initial_epoch= 0, epochs= epochs, ask_epoch= ask_epoch )]

#model_incep.summary()
#for i, layer in enumerate(model_incep.layers):
    #print(f"Layer {i+1}: {layer.name}")
#tf.keras.utils.plot_model(model_incep, show_shapes=True)

history = model_incep.fit(x= train_gen, epochs= epochs, verbose= 0, callbacks= callbacks,
                    validation_data= valid_gen, validation_steps= None, shuffle= False,
                    initial_epoch= 0)
plot_training(history)

preds = model_incep.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

target_names = ['daisy','dandelion','rose','sunflower','tulip']
# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)
plot_confusion_matrix(cm= cm, classes= target_names, title = 'Confusion Matrix')
# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= target_names))

import requests
import matplotlib.pyplot as plt
from PIL import Image
def display_image_from_url(url, figsize=(16, 16)):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    img = Image.open(response.raw)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
image_url = "https://i.imgur.com/s4rx6eC.jpg"
display_image_from_url(image_url, figsize=(16, 16))

import requests
import matplotlib.pyplot as plt
from PIL import Image
def display_image_from_url(url, figsize=(16, 16)):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    img = Image.open(response.raw)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
image_url = "https://i.imgur.com/Upa397Y.png"
display_image_from_url(image_url, figsize=(16, 16))