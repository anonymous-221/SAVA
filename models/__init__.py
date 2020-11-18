import sys
import os.path as osp
import numpy as np
import tensorflow as tf
import os
import i3d
# Add the kaffe module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), './')))
from tensorflow.python.keras.backend import set_session
from inception.inception import inception_model
from googlenet import GoogleNet
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

#from vgg import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from alexnet import AlexNet
from caffenet import CaffeNet
from nin import NiN
from resnet import ResNet50, ResNet101, ResNet152
from lstm import DynamicRNN
from lstm import AveragePooling
from keras_video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier
from tensorflow.keras.models import Model,load_model



def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls


@auto_str
class DataSpec(object):
    '''Input data specifications for an ImageNet model.'''

    def __init__(self,
                 batch_size,
                 scale_size,
                 crop_size,
                 isotropic,
                 channels=3,
                 rescale=[0.0, 255.0],
                 mean=np.array([104., 117., 124.]),
                 bgr=True):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The image should be scaled to this size first during preprocessing
        self.scale_size = scale_size
        # Whether the model expects the rescaling to be isotropic
        self.isotropic = isotropic
        # A square crop of this dimension is expected by this model
        self.crop_size = crop_size
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # The values below are ordered BGR, as many Caffe models are trained in this order.
        # Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
        # However, using just the per-channel mean values instead doesn't
        # affect things too much.
        self.mean = mean
        # Whether this model expects images to be in BGR order
        self.expects_bgr = bgr
        self.rescale = rescale


def alexnet_spec(batch_size=500):
    """Parameters used by AlexNet and its variants."""
    return DataSpec(
        batch_size=batch_size,
        scale_size=256,
        crop_size=227,
        isotropic=False)


def ensemble_spec():
    return DataSpec(
        batch_size=1,
        scale_size=224,
        crop_size=224,
        isotropic=False)


def inception_spec(batch_size=25, crop_size=299, bgr=False):
    """Parameters used by Inception and its variants."""
    return DataSpec(batch_size=batch_size,
                    scale_size=crop_size,
                    crop_size=crop_size,
                    isotropic=False,
                    bgr=bgr,
                    rescale=[0.0,
                             255.0],
                    mean=np.array([0.,
                                   0.,
                                   0.]))


def std_spec(batch_size, isotropic=True):
    """Parameters commonly used by "post-AlexNet" architectures."""
    return DataSpec(
        batch_size=batch_size,
        scale_size=256,
        crop_size=224,
        isotropic=False)


# Collection of sample auto-generated models
_LABEL_MAP_PATH = 'kinetics-i3d/data/label_map.txt'
str2Model = {
    "AlexNet": AlexNet,
    "CaffeNet": CaffeNet,
    "GoogleNet": GoogleNet,
    "NiN": NiN,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "VGG16": VGG16}
MODELS = (
    AlexNet,
    CaffeNet,
    GoogleNet,
    NiN,
    ResNet50,
    ResNet101,
    ResNet152,
    VGG16)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
# The recommended batch size is based on a Titan X (12GB).
MODEL_DATA_SPECS = {
    "AlexNet": alexnet_spec(),
    "CaffeNet": alexnet_spec(),
    "GoogleNet": std_spec(batch_size=25, isotropic=False),
    "ResNet50": std_spec(batch_size=25),
    "ResNet101": std_spec(batch_size=25),
    "ResNet152": std_spec(batch_size=25),
    "NiN": std_spec(batch_size=500),
    "VGG16": std_spec(batch_size=25),
    "Inception": inception_spec(batch_size=25, crop_size=299),
    "Inception2": inception_spec(batch_size=25, crop_size=224),
    "ensemble": ensemble_spec(),
    "i3d_inception":inception_spec(batch_size=25, crop_size=224),
    "vgg16_lstm" : inception_spec(batch_size=25, crop_size=224),
    "I3V":inception_spec(batch_size=25, crop_size=224),
    "LSTM":inception_spec(batch_size=25, crop_size=224),
    "i3d": inception_spec(batch_size=25, crop_size=224)
 
}

MODEL_PATHES = {
    "AlexNet": "models/AlexNet.npy",
    "CaffeNet": "models/CaffeNet.npy",
    "GoogleNet": "models/GoogleNet.npy",
    "ResNet50": "models/ResNet50.npy",
    "ResNet101": "models/ResNet101.npy",
    "ResNet152": "models/ResNet152.npy",
    "VGG16": "models/VGG16.npy"
}

CKPT_PATHES = {
    
    "UCF101_Inception2": "checkpoints/UCF101/Inception2/Inception.ckpt",
    "UCF101_i3d_inception":"checkpoints/UCF101/I3D/ucf101_rgb_0.946_model-44520",
    "HMDB51_i3d_inception":"checkpoints/HMDB51/I3D/models_0.800_model-10000",
    
    "HMDB51_I3V":"checkpoints/HMDB51/I3V/inception.019-2.16.hdf5",
    "UCF101_I3V": "checkpoints/UCF101/I3V/inception.014-1.04.hdf5",
    "i3d_imagenet": "data/checkpoints/rgb_imagenet/model.ckpt"
}
INPUT_DIMS = {
    "resnet" : 50,
    "i3d": 250,
    "c3d": 16,
    "tsn": 3
    
}


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def get_model_path(model_name):
    return MODEL_PATHES[model_name]


def get_data_spec(model_name):
    """Returns the data specifications for the given network."""
    return MODEL_DATA_SPECS[model_name]


def get_model_i3(class_no):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(class_no, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_model(sess, input_node, model_name,data_set_name, device=None):
    if model_name in ["GoogleNet","ResNet50","ResNet101","ResNet152","VGG16","Inception","Inception2"]:
    
        _,seq_len, _, _, _ = input_node.get_shape()
        seq_len = seq_len.value
        start_variable_set = set(tf.all_variables())
        rescaled_input_node = tf.image.resize_bilinear(input_node[:, 0, :, :, :], [299, 299])
        end_node1 = get_inception2(rescaled_input_node)
        end_node1 = tf.expand_dims(end_node1, 1)
        lstm_input = end_node1
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for ii in range(seq_len-1):
                rescaled_input_node = tf.image.resize_bilinear(input_node[:, -(ii+1), :, :, :], [299, 299])
                end_node = get_inception2(rescaled_input_node)
                end_node = tf.expand_dims(end_node, 1)
                lstm_input = tf.concat([end_node,lstm_input], 1)
        lstm_input = tf.concat([end_node1,lstm_input[:,0:-1,:]],1)
        end_variable_set = set(tf.all_variables())
        variable_set = end_variable_set - start_variable_set
        print ('Loading prarameters')
        saver = tf.train.Saver(variable_set)
        ckpt_dir = CKPT_PATHES[data_set_name+'_'+model_name]
        print ('Checkpoint dir', ckpt_dir)
        saver.restore(sess, ckpt_dir)
        print ('Loaded prarameters')
        lstm_model = DynamicRNN(lstm_input, num_classes=101, cell_size=512, use_lstm=True)
        end1_variable_set = set(tf.all_variables())
        lstm_set = end1_variable_set - end_variable_set
        print ('Loading lstm parameters')
        saver1 = tf.train.Saver(lstm_set)
        ckpt_lstm = 'checkpoints/'+data_set_name +'/LSTM/ckpt-3000'
        print ('Checkpoint dir', ckpt_lstm)
        saver1.restore(sess, ckpt_lstm)
        print ('Loaded lstm parameters')
        #return lstm_model.logit, variable_set.union(lstm_set), lstm_model.prediction, lstm_input, lstm_model.node
        return lstm_model.logit, variable_set.union(lstm_set), lstm_model.prediction
    elif model_name == 'LSTM':
        _,seq_len, _, _, _ = input_node.get_shape()
        seq_len = seq_len.value
        graph = tf.get_default_graph()
        set_session(sess)
        start_variable_set = set(tf.all_variables())
        
        base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )
        extractor = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )
        
        rescaled_input_node = tf.image.resize_bilinear(input_node[:, 0, :, :, :], [299, 299])
        end_node1 = extractor(rescaled_input_node)
        end_node1 = tf.expand_dims(end_node1, 1)
        lstm_input = end_node1
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for ii in range(seq_len-1):
                rescaled_input_node = tf.image.resize_bilinear(input_node[:, ii+1, :, :, :], [299, 299])
                end_node = extractor (rescaled_input_node)
                end_node = tf.expand_dims(end_node, 1)
                lstm_input = tf.concat([end_node,lstm_input], 1)
        #lstm_input = tf.concat([end_node1,lstm_input[:,0:-1,:]],1)
        
        print ('Loading prarameters')
        
        saved_model = 'checkpoints/HMDB51/CNN_LSTM/lstm-features.008-2.567.hdf5'
        model = load_model(saved_model)
        optimizer = Adam(lr=1e-5, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=['accuracy'])
        end_variable_set = set(tf.all_variables())
        variable_set = end_variable_set - start_variable_set
        with graph.as_default():
            set_session(sess)
            logits = model(lstm_input)
     
        prediction = tf.argmax(logits,axis = 1)
        return logits,variable_set, prediction
    
    elif model_name =='I3V':
        
        if data_set_name =='UCF101':
            class_no = 101
        else:
            class_no = 51
        batch_size,seq_len, _, _, _ = input_node.get_shape()
        seq_len = seq_len.value
        graph = tf.get_default_graph()
        set_session(sess)
        start_variable_set = set(tf.all_variables())
        model =get_model_i3(class_no)
        model.load_weights(CKPT_PATHES[data_set_name+'_'+model_name])
        #model = load_model(CKPT_PATHES[data_set_name+'_'+model_name])
        
        rescaled_input_node = tf.image.resize_bilinear(input_node[:, 0, :, :, :], [299, 299])
        
        inputs = rescaled_input_node
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for ii in range(seq_len-1):
                rescaled_input_node = tf.image.resize_bilinear(input_node[:, ii+1, :, :, :], [299, 299])
                
                inputs = tf.concat([rescaled_input_node,inputs], 1)
        #lstm_input = tf.concat([end_node1,lstm_input[:,0:-1,:]],1)
        end_variable_set = set(tf.all_variables())
        variable_set = end_variable_set - start_variable_set
        #rescaled_input_node = tf.image.resize_bilinear(input_node, [batch_size,seq_len,299, 299,3])
        with graph.as_default():
            set_session(sess)
            logits = model(inputs)
        
        print(logits)
        prediction = tf.argmax(logits,axis=1)
        return logits,variable_set, prediction
        
        
    else:
        #kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
        #rescaled_input_node = tf.image.resize_bilinear(input_node[:, 0, :, :, :], [224, 224])
        start_variable_set = set(tf.all_variables())
        with tf.variable_scope('RGB'):
            if data_set_name =="UCF101":
                class_num =101

            #else:
                #class_num =51

                rgb_model = i3d.InceptionI3d(
                    400, spatial_squeeze=True, final_endpoint='Logits')
                rgb_logits, _ = rgb_model(input_node, is_training=False, dropout_keep_prob=1.0)
                rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
                rgb_fc_out = tf.layers.dense(rgb_logits_dropout, class_num, use_bias=True)
            else:
                class_num =51

                rgb_model = i3d.InceptionI3d(
                    class_num, spatial_squeeze=True, final_endpoint='Logits')
                rgb_logits, _ = rgb_model(input_node, is_training=False, dropout_keep_prob=1.0)
                rgb_fc_out = rgb_logits
               # rgb_fc_out = tf.nn.softmax(rgb_logits)
                #rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
                #rgb_fc_out = tf.layers.dense(
                #rgb_logits_dropout, class_num, use_bias=True)

        rgb_variable_map = {}
        
        #rgb_saver = tf.train.Saver(var_list=rgb_variable_set,reshape=True)
        variable_map = {}
        
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == 'RGB':
                variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
        
        '''
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '').replace('Conv3d','Conv2d').replace('conv_3d/w','weights').replace('conv_3d/b','biases').replace('RGB/inception_i3d', 'InceptionV1').replace('batch_norm','BatchNorm')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map,reshape=True)
        '''
        end_variable_set = set(tf.all_variables())
        rgb_variable_set = end_variable_set - start_variable_set
        
        #print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, CKPT_PATHES[data_set_name+'_'+model_name])
        print("load complete!")
        #rgb_saver.restore(sess, CKPT_PATHES[data_set_name+'_'+model_name])
        print ('Checkpoint dir', CKPT_PATHES[data_set_name+'_'+model_name])
        
        
        prediction = tf.argmax(rgb_fc_out, axis=1)
        
            
        return rgb_fc_out ,rgb_variable_set, prediction


def get_model3(sess, input_node, model_name, device=None):
    print ('Getting model def', model_name)
    start_variable_set = set(tf.all_variables())
    if model_name == 'Inception':
        end_node = get_inception(input_node)
    elif model_name == 'Inception2':
        # swap_rgb = tf.reverse(input_node, [False, False, True])
        rescaled_input_node = tf.image.resize_bilinear(input_node, [299, 299])
        end_node = get_inception(rescaled_input_node)
    else:
        all_models = MODELS
        net_class = [
            model for model in all_models if model.__name__ == model_name][0]
        net = net_class({'data': input_node})
        end_node = net.get_output()
        end_node2 = tf.argmax(end_node,1)

    end_variable_set = set(tf.all_variables())
    variable_set = end_variable_set - start_variable_set
    print ('Loading prarameters')
    saver = tf.train.Saver(variable_set)
    ckpt_dir = CKPT_PATHES[model_name]
    print ('Checkpoint dir', ckpt_dir)
    saver.restore(sess, ckpt_dir)
    print ('Loaded prarameters')
    return end_node, variable_set, end_node2

def get_model2(sess, input_node, model_name, device=None):
    print ('Getting model def', model_name)
    start_variable_set = set(tf.all_variables())
    if model_name == 'Inception':
        end_node, end_node2 = get_inception2(input_node)
    elif model_name == 'Inception2':
        rescaled_input_node = tf.image.resize_bilinear(input_node, [299, 299])
        end_node, end_node2 = get_inception2(rescaled_input_node)
    else:
        all_models = MODELS
        net_class = [
            model for model in all_models if model.__name__ == model_name][0]
        net = net_class({'data': input_node})
        end_node = net.get_output()
        if model_name == 'VGG16':
            end_node2 = net.layers['fc8']
        elif model_name == 'GoogleNet':
            end_node2 = net.layers['loss3_classifier']
        else:
            end_node2 = net.layers['fc1000']
    end_variable_set = set(tf.all_variables())
    variable_set = end_variable_set - start_variable_set
    print ('Loading prarameters')
    saver = tf.train.Saver(variable_set)
    ckpt_dir = CKPT_PATHES[model_name]
    print ('Checkpoint dir', ckpt_dir)
    saver.restore(sess, ckpt_dir)
    print ('Loaded prarameters')
    return end_node, end_node2


def get_inception(images):
    return inception_model.inference(images=images, num_classes=1000 + 1)


def get_inception2(images):
    return inception_model.inference2(images=images, num_classes=1000 + 1)


def inception_load_parameters(sess, var_list=None):
    saver = tf.train.Saver(var_list)
    ckpt_dir = 'models/inception/checkpoint'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print (ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found')
        return


def save_all_as_checkponits():
    # Now this function only save ckpt file for GoogleNet,
    # TODO: Save ckpt for all 9 networks
    for model_name in ["GoogleNet"]:
        if not tf.train.checkpoint_exists(CKPT_PATHES[model_name]):
            print ("Checkpoint for " + model_name + " has not been created yet, creating checkpoint...")
            spec = get_data_spec(model_name)
            input_node = tf.placeholder(
                tf.float32,
                shape=(
                    None,
                    spec.crop_size,
                    spec.crop_size,
                    spec.channels))
            net = str2Model[model_name]({'data': input_node})
            with tf.Session() as sesh:
                net.load(get_model_path(model_name), sesh)
                saver = tf.train.Saver()
                save_path = saver.save(sesh, CKPT_PATHES[model_name])
                print(
                    model_name +
                    " Model checkpoint saved in file: %s" %
                    save_path)


#save_all_as_checkponits()
