

import tensorflow as tf


from tensorflow.python.platform import tf_logging as logging
import DenseNet.nets.densenet as densenet
import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
import research.slim.nets.inception_resnet_v2 as inception
from utils.gen_tfrec import load_batch, get_dataset, load_batch_dense, load_batch_estimator

import os
import sys
from yaml import load, dump
slim = tf.contrib.slim

config_file = sys.argv[1] or os.path.join(os.getcwd(), "config.yaml")

#Open and read the yaml file:
stream = open(os.path.join(config_file))
data = load(stream)

print("loaded config file:\n")
print(data)
print("--- end ---\n")

#=======Dataset Informations=======#
#==================================#
dataset_dir = data["dataset_dir"]
train_dir = data["train_dir"]
summary_dir = os.path.join(train_dir , "summary")
gpu_p = data["gpu_p"]
#Emplacement du checkpoint file
checkpoint_dir= data["checkpoint_dir"]
checkpoint_file = os.path.join(checkpoint_dir, "mobilenet_v2_1.4_224.ckpt")
ckpt_state = tf.train.get_checkpoint_state(train_dir)
image_size = data["image_size"]
#Nombre de classes à prédire
file_pattern = data["file_pattern"]
file_pattern_for_counting = data["file_pattern_for_counting"]

num_samples = data["num_samples"]
#Création d'un dictionnaire pour reférer à chaque label
#MURA Labels
names_to_labels = data["names_to_labels"]
labels_to_names = data["labels_to_names"]
#==================================#
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = data["num_epochs"]
#State your batch size
batch_size = data["batch_size"]
#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = data["initial_learning_rate"]
#Decay factor
learning_rate_decay_factor = data["learning_rate_decay_factor"]
num_epochs_before_decay = data["num_epochs_before_decay"]
#Calculus of batches/epoch, number of steps after decay learning rate
num_batches_per_epoch = int(num_samples / batch_size)
#num_batches = num_steps for one epcoh
decay_steps = int(num_epochs_before_decay * num_batches_per_epoch)
#==================================#
#Create log_dir:
if not os.path.exists(train_dir):
    os.mkdir(os.path.join(train_dir))
if not os.path.exists(summary_dir):
    os.mkdir(os.path.join(summary_dir))

#===================================================================== Training ===========================================================================#
#Adding the graph:
#Set the verbosity to INFO level
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.DEBUG)

def input_fn(mode, dataset_dir,file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("dataset"):
        dataset = get_dataset("train" if train_mode else "eval",
                                        dataset_dir, file_pattern=file_pattern,
                                        file_pattern_for_counting=file_pattern_for_counting,
                                        labels_to_name=labels_to_name)
    with tf.name_scope("load_data"):
        dataset = load_batch_estimator(dataset, batch_size, image_size, image_size,
                                                        shuffle=train_mode, is_training=train_mode)
    return dataset 

def model_fn(features, mode):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    tf.summary.image("images",features['image/encoded'])
    #Create the model inference
    with slim.arg_scope(mobilenet_v2.training_scope(is_training=train_mode, weight_decay=1e-4, stddev=5e-2, bn_decay=0.99)):
            #TODO: Check mobilenet_v1 module, var "excluding
            logits, _ = mobilenet_v2.mobilenet(features['image/encoded'],depth_multiplier=1.4, num_classes = len(labels_to_names))
    excluding = ['MobilenetV2/Logits']
    variables_to_restore = slim.get_variables_to_restore(exclude=excluding)
    if (not ckpt_state) and checkpoint_file and train_mode:
        variables_to_restore = variables_to_restore[1:]
        tf.train.init_from_checkpoint(checkpoint_file, 
                            {v.name.split(':')[0]: v for v in variables_to_restore})

    predicted_classes = tf.argmax(logits, axis=1)
    labels = tf.cast(features["image/class/id"], tf.int32) 
    
        #Defining losses and regulization ops:
    with tf.name_scope("loss_op"):
        loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
        total_loss = tf.losses.get_total_loss() #obtain the regularization losses as well
    #FIXME: Replace classifier function (sigmoid / softmax)
    if mode != tf.estimator.ModeKeys.PREDICT:
        metrics = {
        'Accuracy': tf.metrics.accuracy(labels, predicted_classes, name="acc_op"),
        'Precision': tf.metrics.precision(labels, predicted_classes, name="precision_op"),
        'Recall': tf.metrics.recall(labels, predicted_classes, name="recall_op"),
        'Acc_Class': tf.metrics.mean_per_class_accuracy(labels, predicted_classes,len(labels_to_names), name="per_class_acc")
        }
        for name, value in metrics.items():
            items_list = value[1].get_shape().as_list()
            if len(items_list) != 0:
                for k in range(items_list[0]):
                    tf.summary.scalar(name+"_"+labels_to_names[str(k)], value[1][k])
            else:
                tf.summary.scalar(name, value[1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        #Create the global step for monitoring the learning_rate and training:
        global_step = tf.train.get_or_create_global_step()
        with tf.name_scope("learning_rate"):    
            lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate = learning_rate_decay_factor,
                                    staircase=True)
            tf.summary.scalar('learning_rate', lr)
        #Define Optimizer with decay learning rate:
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)      
            train_op = slim.learning.create_train_op(total_loss,optimizer,
                                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
    
    #For Evaluation Mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    #For Predict/Inference Mode:
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes':predicted_classes,
            'probabilities': tf.nn.softmax(logits, name="Softmax")
            }      
        export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode,predictions=predictions,
                                            export_outputs=export_outputs)
def main():
    #Define the checkpoint state to determine initialization: from pre-trained weigths or recovery
           
    #Define max steps:
    max_step = num_epochs*num_batches_per_epoch
    #Define configuration non-distributed work:
    run_config = tf.estimator.RunConfig(model_dir=train_dir, save_checkpoints_steps=num_batches_per_epoch,keep_checkpoint_max=num_epochs)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.TRAIN,
                                                dataset_dir,file_pattern,
                                                file_pattern_for_counting, names_to_labels,
                                                batch_size, image_size),max_steps=max_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.EVAL,
                                                    dataset_dir, file_pattern,
                                                    file_pattern_for_counting, names_to_labels,
                                                    batch_size,image_size))
    work = tf.estimator.Estimator(model_fn = model_fn,
                                    model_dir=train_dir,
                                    config=run_config)
       
    tf.estimator.train_and_evaluate(work, train_spec, eval_spec)
if __name__ == '__main__':
    main()