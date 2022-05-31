from model import vgg_19
from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow import layers
from sklearn.model_selection import train_test_split
import torch

#Checking for GPUs
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print("Number of available GPU(s) %d." % torch.cuda.device_count())

    print("GPU Name: ", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

print("\n---------------------------------------\n")  

#Set up Comet ML 
experiment = Experiment(
    api_key="d1doFSsP6roSDbhpchOqHAc8G",
    project_name="VGG_19 Experiments",
    workspace="reniven",
    log_code = True,
    auto_param_logging=True
)

#Load data and data augmentation
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2)

print("Training Data: ", X_train.shape)
print("Validation Data: ", X_val.shape)
print("Test Data: ", X_test.shape)

keras.backend.clear_session()

#Create training and testing variables
batch_size = 124
epochs = 300

#Create Piecewise constant learning rate scheduler
num_steps = len(X_train)/batch_size
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    [150*num_steps, 250*num_steps],
    [0.1, 0.01, 0.001]
    )

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("\nNumber of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    layer_info = ((2, 64), (2, 128), (2, 256), (4, 512), (4, 512))
    vgg_net = vgg_19(layer_info, 224, 224, 3, 1000, 0.0001, False)
    vgg_net.summary()
    vgg_net.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate_fn, momentum=0.9),
    loss = 'categorical_crossentropy')

checkpoint_cb = keras.callbacks.ModelCheckpoint("IMDB-GRU.h5", save_best_only=True)

#Begin training and logging to Comet ML
with experiment.train():
    history = vgg_net.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (y_train, y_val)
    )

#Begin testing and logging to Comet ML
with experiment.test():

    loss, accuracy = vgg_net.evaluate(X_test, y_test)
    print(loss, accuracy)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }

    experiment.log_metrics(metrics)

vgg_net.save("IMDB-GRU.h5")