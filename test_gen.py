######For one video each a time 
import argparse
import os
import sys
import shutil
import models
import numpy as np
import tensorflow as tf
import scipy.misc
from data import DataSet
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import cv2
import numpy as np
from PIL import Image
from ssim import SSIM
import time
from stn import spatial_transformer_network as transformer
_errstr = "Mode is unknown or incompatible with input array shape."
#import tensorflow_probability as tfp
import stadv
from GA_mask import gen_a,my_sa,sa_tsp,ba_op,ba_op_4
from matplotlib.ticker import FormatStrFormatter
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def affine(img):
    T = np.array([[0.1255,0.5642,20],[1.0041,0,10],[0,0,1]])
    
    height, width, channel = img.shape
    
    affine_img = tf.Variable(tf.zeros(( height, width, 3) ))#创建变换后图像
    src_point = np.array([[0, 0, height-1 ], [0, width-1, 0], [1, 1, 1]])
    
    for x in range( height ):
      
        for y in range( width ):
            src_image_point = [[ x, y,1 ]]
            src_image_point_transpose = np.transpose( src_image_point )
            
            a = np.dot( T, src_image_point_transpose )
            new_x, new_y= a[:2]
            
            tf.assign(affine_img[int(new_x),int(new_y),:] , img[x,y,:])
            
    return affine_img
'''
def rotate(images,angles):
    
    return tf.contrib.image.transform(
    images,
    tf.contrib.image.angles_to_projective_transforms(
        angles, tf.cast(tf.shape(images)[0], tf.float32), tf.cast(tf
            .shape(images)[1], tf.float32)
    ))
'''
def rotate(image, angle, mode='repeat'):
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """
    s = image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [np.floor(x/2) for x in s]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.stack ([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([[0], [1], [2], [3]], [[tf.cos(angle)], [tf.sin(angle)], [-tf.sin(angle)], [tf.cos(angle)]])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.stack ([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    image_channel_list = tf.split(image, s[2], 2)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated
def calc_gradients(
        test_file,
        data_set_name,
        model_name,
        output_file_dir,
        max_iter,
        learning_rate=0.0001,
        targets=None,
        weight_loss2=1,
        data_spec=None,
        batch_size=1,
        total_len=40,
        seq_len = 10):

    """Compute the gradients for the given network and images."""
    spec = data_spec
    if data_set_name =='UCF101':
        class_no =101
    else:
        class_no = 51
    #initial_T =  np.array([[0.1255,0.5642,20],[1.0041,0,10],[0,0,1]])
    modifier = tf.Variable(0.01*np.ones((1, seq_len, spec.crop_size,spec.crop_size,spec.channels),dtype=np.float32))
    blur_para = tf.Variable(0.01*np.ones((1, seq_len, spec.crop_size,spec.crop_size,spec.channels),dtype=np.float32))
    # identity transform
   
    input_image = tf.placeholder(tf.float32, (batch_size, total_len, spec.crop_size, spec.crop_size, spec.channels))
    input_label = tf.placeholder(tf.int32, (batch_size))
    image_to_rotate = tf.placeholder(shape=(1,spec.crop_size, spec.crop_size, spec.channels), dtype=tf.float32)
    angle_to_rotate = tf.placeholder(shape=(6), dtype=tf.float32)
    
    #theta = tf.placeholder(tf.float32,shape=((seq_len)))
    theta = tf.placeholder(shape=(seq_len), dtype=tf.float32)
    flows_var = tf.placeholder(tf.float32,shape=((1,2, spec.crop_size,spec.crop_size)))
    
    rotate_result = stadv.layers.flow_st( transformer(image_to_rotate,angle_to_rotate), flows_var, 'NHWC')
    #flows = tf.placeholder(tf.float32, [seq_len, 2,spec.crop_size, spec.crop_size], name='flows')
    flows = tf.Variable(np.zeros((seq_len,2, spec.crop_size,spec.crop_size),dtype=np.float32))
    tau = tf.placeholder_with_default(
        tf.constant(0., dtype=tf.float32),
        shape=[], name='tau'
    )
    initial = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    
    initial_pi = np.array([
        [np.cos(np.pi/5), -np.sin(np.pi/5), 0.1],
        [np.sin(np.pi/5), np.cos(np.pi/5), 0.1]
    ])
    initial_pi = tf.reshape(initial_pi.astype('float32').flatten(),(6,1))
    initial = tf.reshape(initial.astype('float32').flatten(),(6,1))
    x_trans = tf.reshape(input_image,[-1,spec.crop_size*spec.crop_size*spec.channels])
    # localization network
    #W_fc1 = tf.Variable(initial_value=initial_pi)
    b_fc1 = tf.Variable(initial_value=initial)
    
    
    
    b_1 = tf.Variable(np.zeros(shape = (seq_len)),dtype=np.float32)
    b_2 = tf.Variable(np.zeros(shape = (seq_len)),dtype=np.float32)
    
    #W_fc1 = [[tf.cos(theta),-tf.sin(theta),0.2*tf.tanh(b_1)],[tf.sin(theta),tf.cos(theta),0.2*tf.tanh(b_2)]]
    
    #W_fc1 = tf.reshape( W_fc1,(6,1))
    '''
    b_fc_loc1 = tf.Variable(tf.random_normal([10], mean=0.0, stddev=0.01))
    W_fc_loc2 = tf.Variable(tf.zeros([10, 6]))
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x_trans, W_fc1) + b_fc_loc1)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, 1)
    '''
   
    
    #angle = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc2)
    #angle = tf.nn.tanh(tf.matmul(x_trans, W_fc1)+ b_fc1)
    #angle = tf.expand_dims(b_fc1,0)
    
    #angle = tf.Variable(tf.random.uniform(shape = (seq_len,), minval = -np.pi / 4, maxval = np.pi / 4))
    #angle = tf.Variable(tf.random.uniform(shape = (seq_len,), minval = -10, maxval = 10))
    #angle = tf.Variable(0.1*np.ones(( seq_len,),dtype=np.float32))
    
    # temporal mask, 1 indicates the selected frame
    #indicator = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0]
    indicator = tf.placeholder(tf.float32,shape=(seq_len))
    #theta = theta * indicator
    #indicator = [0,0,1,1,1,0,0,0,0,0]
    #indicator = np.reshape(indicate,(1,10))
    #indicator = tf.Variable(np.random.randint(1,4,(1,seq_len)),
    #dtype=np.int64)
    #where(t > 0, tf.ones((seq_len,1)), tf.zeros((seq_len,1)))
    #indicator= np.reshape(np.repeat(
       #indicator, (6), axis=0),(seq_len,6))
    #indicator = tf.Variable(tf.random.uniform(shape = (1,seq_len), minval = 0,  maxval = 3),constraint = lambda t: tf.where(t > 0, tf.ones((1,seq_len)), tf.zeros((1,seq_len))))

    # spatial transformer layer
    #h_trans = transformer(x, h_fc1)
    #true_image = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+transformer(tf.expand_dims(input_image[0,0,:,:,:]*255.0,0),angle[0]), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
    #true_image = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+input_image[0,0,:,:,:]*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
    #ind = tf.reshape(angle,(seq_len,6))*indicator
    #true_image = tf.minimum(tf.maximum(transformer(tf.expand_dims(input_image*255.0,0),angle), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
    #angle = tf.transpose(tf.matmul(W_fc1,tf.cast(indicator,tf.float32))) + tf.transpose(tf.matmul(b_fc1,tf.cast(1-indicator,tf.float32)))
    '''
    indicat= tf.repeat(
        indicator, (2), axis=0
    )
    indicat= tf.repeat(
        indicat, (spec.crop_size), axis=0
    )
    indicat= tf.repeat(
        indicat, (spec.crop_size), axis=0
    )
    indicate= tf.repeat(
        indicator, (spec.crop_size), axis=0
    )
    indicate= tf.repeat(
        indicate, (spec.crop_size), axis=0
    )
    indicate= tf.reshape(tf.repeat(
        indicate, (3), axis=0
    ),(seq_len, spec.crop_size,spec.crop_size,spec.channels))
    indicat = tf.reshape(indicat,(seq_len, 2,spec.crop_size, spec.crop_size))
    '''
    #flows = indicat * flows
    #modifier = tf.cast(indicate,tf.float32) * modifier
    
    
    for ll in range(seq_len):
        #if indicator[ll] == 1:
            #the = theta[ll]*indicator[ll]
            #angle = [[tf.cos(the),-tf.sin(the),0.1*indicator[ll]],[tf.sin(the),tf.cos(the),0.1*indicator[ll]]]
        
            #rotate_img = transformer(tf.expand_dims(input_image[0,ll,:,:,:],0),angle)
        #else:
            #rotate_img = tf.expand_dims(input_image[0,ll,:,:,:],0)
        #perturbed_images = tf.minimum(tf.maximum(stadv.layers.flow_st( transformer(tf.expand_dims(input_image[0,ll,:,:,:],0)*255.0,angle), flows[ll]*indicator[ll], 'NHWC'), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        perturbed_images = tf.minimum(tf.maximum(stadv.layers.flow_st( tf.expand_dims(input_image[0,ll,:,:,:],0)*255.0, flows[ll]*indicator[ll], 'NHWC'), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        mask_temp = tf.minimum(tf.maximum(modifier[0,ll,:,:,:]*indicator[ll]+perturbed_images[0]*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        mask_temp = tf.expand_dims(mask_temp , 0)
        if ll==0:
            true_image = mask_temp
        else:
            #mask_temp = input_image[0,ll+1,:,:,:]
            #mask_temp = tf.expand_dims(mask_temp,0)
            true_image = tf.concat([true_image, mask_temp],0)
    if seq_len < total_len:
        true_image  = tf.concat([true_image, input_image[0,seq_len:total_len,:,:,:]],0)
    true_image = tf.expand_dims(true_image, 0)
    
    for kk in range(batch_size-1):
        #true_image_temp = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+transformer(tf.expand_dims(input_image[kk+1,0,:,:,:]*255.0,0),angle[0]), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        true_image_temp = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+input_image[kk+1,0,:,:,:]*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        true_image_temp = tf.expand_dims(true_image_temp, 0)
        for ll in range(seq_len-1):
            if indicator[ll+1] == 1:
               mask_temp = tf.minimum(tf.maximum(modifier[0,ll+1,:,:,:]+transformer(tf.expand_dims([input_image[kk+1,ll+1,:,:,:]*255.0],0),angle[0]), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
            else:
               mask_temp = input_image[kk+1,ll+1,:,:,:]
               mask_temp = tf.expand_dims(mask_temp,0)
            true_image_temp = tf.concat([true_image_temp, mask_temp],0)
        true_image_temp = tf.expand_dims(true_image_temp, 0)

        true_image = tf.concat([true_image, true_image_temp],0)
    
    loss2_l12 = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(true_image-input_image), axis=[0, 2, 3, 4])))
    #loss2 = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(true_image-input_image), axis=[0, 2, 3, 4])))
    loss2 =  tf.reduce_sum(1-tf.image.ssim_multiscale(true_image, input_image, max_val=1.0))
    #loss2 = 1.0 - tf.reduce_mean(SSIM(true_image).cw_ssim_value(input_image))
    
    norm_frame = tf.reduce_mean(tf.abs(modifier), axis=[2,3,4])
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #input_probs, _, input_pre_label,input_ince_output, input_pre_node = models.get_model(sess, input_image, model_name, False)
    probs, variable_set, pre_label = models.get_model(sess, true_image, model_name,data_set_name, False)
    true_label_prob = tf.reduce_sum(probs*tf.one_hot(input_label,class_no),[1])
    if model_name == 'i3d_inception':
          true_label_prob = tf.reduce_mean(probs*tf.one_hot(input_label,class_no))
       #input_true_label_prob = tf.reduce_sum(input_probs*tf.one_hot(input_label,101),[1])
      
    #input_true_label_prob = tf.reduce_sum(input_probs*tf.one_hot(input_label,101),[1])
    if targets is None:
        #loss1 = tf.maximum(0.0,true_label_prob)
        loss1 = -tf.log(1 - true_label_prob + 1e-6)
    else:
        loss1 = -tf.log(true_label_prob + 1e-6)
    loss1 = tf.reduce_sum(loss1)
    
    
    loss = loss1 + weight_loss2 * (loss2 +loss2_l12) + tf.reduce_mean(tf.abs(flows))
   
    
    grad_op = tf.gradients(loss,theta)
    #很不稳定变化很大忽上忽下
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #一直下降就是很慢
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
    #grad = tf.gradients(loss,indicator )
    #变化大
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    #变化大稍微好一点
    #optimizer= tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9, use_nesterov=True)
    #optimizer=tf.train.AdagradOptimizer(learning_rate=1,initial_accumulator_value=0.01)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #忽上忽下变化大
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=1, decay=0.9, momentum=0.9)
    #忽上忽下变化大
    #optimizer =tf.train.RMSPropOptimizer(learning_rate=1, decay=0.9)
    print('optimizer.minimize....')
    
    train = optimizer.minimize(loss, var_list=[modifier,flows])
    Train = optimizer.minimize(loss1, var_list=[modifier,flows])
    # initiallize all uninitialized varibales
    init_varibale_list = set(tf.all_variables()) - variable_set
    
    sess.run(tf.initialize_variables(init_varibale_list))

    data = DataSet(data_set=data_set_name,test_list=test_file, seq_length=seq_len,image_shape=(spec.crop_size, spec.crop_size, spec.channels))
    print('data loaded')
    all_names = []
    all_images = []
    all_labels = []
    output_names = []
    def_len = seq_len
    for video in data.test_data:
        frames,f_name = data.get_frames_for_sample(data_set_name,video)
        if len(frames) < def_len:
           continue
        frames = data.rescale_list(frames, def_len)
        frames_data = data.build_image_sequence(frames)
        all_images.append(frames_data)
        label, hot_labels = data.get_class_one_hot(video[1])
        all_labels.append(label)
        all_names.append(f_name)
        output_names.append(frames)
    total = len(all_names)
    all_indices = range(total)
    num_batch = int(total/batch_size)
    f = open("rotate_ssim_hcm.txt", "a+")
    print('process data length:', num_batch,file=f)

    correct_ori = 0
    correct_noi = 0
    tot_image = 0
    adv = 0
    sess.run(tf.initialize_variables(init_varibale_list))
    for ii in range(num_batch):
        images = all_images[ii*batch_size : (ii+1)*batch_size]
        names = all_names[ii*batch_size : (ii+1)*batch_size]
        labels = all_labels[ii*batch_size : (ii+1)*batch_size]
        indices = all_indices[ii*batch_size : (ii+1)*batch_size]
        output_name = output_names[ii*batch_size : (ii+1)*batch_size]
        print('------------------prediction for clean video-------------------')
        print('---video-level prediction---')
        
        for xx in range(len(indices)):
            print(names[xx],'label:', labels[xx], 'indice:',indices[xx], 'size:', len(images[xx]), len(images[xx][0]), len(images[xx][0][0]), len(images[xx][0][0][0]))
        
        sess.run(tf.initialize_variables(init_varibale_list))
        
        if targets is not None:
            labels = [targets[e] for e in names]
        
        
        #feed_dict = {input_image: images[0:seq_len], input_label: labels,tau: 0.05,flows:null_flows,indicator:indicator_ini,theta:np.zeros((seq_len))}
        indicator_ini =  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        feed_dict = {input_image: images[0:total_len], input_label: labels,tau: 0.05,indicator:indicator_ini,theta:np.zeros((seq_len))}
        var_loss, true_prob, var_loss1, var_loss2, var_l12loss,var_pre= sess.run((loss, true_label_prob, loss1, loss2,loss2_l12, pre_label), feed_dict=feed_dict)
        
        correct_pre = correct_ori
        for xx in range(len(indices)):
           if labels[xx] == var_pre[xx]:
              correct_ori += 1

        tot_image += 1
        print('Start!')
        min_loss = var_loss
        last_min = -1
        print('---frame-wise prediction---')
        #print('node_label:', var_node, 'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib', true_prob)
        print('label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib', true_prob,'var_l12loss',var_l12loss)
        # record numer of iteration
        tot_iter = 0
        
        if correct_pre == correct_ori:#if model predict is wrong
           ii += 1
           continue
        if true_prob ==1.0:
            ii +=1
            correct_noi +=1
            continue
       
        print('------------------prediction for adversarial video-------------------')
        Test_mode = True
        ge_time =time.time()
        theta_in = np.ones((seq_len))*0.5
        
        index = ba_op(train,init_varibale_list,true_label_prob,seq_len,indicator,f,feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05,theta : theta_in},sess=sess)
        
        print(index)
        mask = np.zeros((seq_len))
        mask[index] =1
        
        #mask = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #mask=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        '''
        ind1,ind2,ind3,ind4 = ba_op_4(train,init_varibale_list,true_label_prob,seq_len,indicator,f,feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05},sess=sess)
        mask = np.zeros((seq_len))
        for i in [ind1,ind2,ind3,ind4]:
            mask[i] = 1
        '''
        print(mask)
        sess.run(tf.initialize_variables(init_varibale_list))
        #sa = my_sa(loss_mask, seq_len,indicator,theta,feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05},sess=sess)
        #Updat GA for angle and Mask
        '''
        ga = gen_a(loss_mask, seq_len,indicator,theta,ga_extra_kwargs={'max_iter':50},feed_dict={input_image: images[0:total_len], input_label: labels, tau: 0.05},
            sess=sess)
        
        
        print('best_mask',ga['mask'],'angle',ga['angle'],'Loss',ga['loss'],'time',time.time()-ge_time,file=f)
        feed_dict = {input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:ga['mask'],theta:ga['angle']}
        '''
        '''
        ga = gen_am(loss_mask, seq_len,indicator,ga_extra_kwargs={'max_iter':20},feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05,theta:np.random.normal(0,np.pi/3,(seq_len))},
            sess=sess)
        
        
        print('best_mask',ga['mask'],'Loss',ga['loss'],'time',time.time()-ge_time,file=f)
        '''
        
        
        '''
        sa = sa_tsp(loss_mask, seq_len,indicator,feed_dict={input_image: images[0:total_len], input_label: labels, tau: 0.05,theta:np.random.normal(0,np.pi/3,(seq_len))},
        sess=sess)
        print('best_mask',sa['mask'],'Loss',sa['loss'],'time',time.time()-ge_time)
        fig, ax = plt.subplots()
        best_points_ = np.concatenate([sa['mask'], [sa['mask'][0]]])
        ax[0].plot(sa.best_y_history)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Distance")
        plt.savefig('./test.jpg')
        plt.show()
        '''
        
        '''
        results_theta = stadv.optimization.lbfgs(
            loss,
            theta,
            # random initial guess for the flow
            flows_x0=np.random.random_sample((seq_len)),
            feed_dict={input_image: images[0:total_len], input_label: labels, tau: 0.05,indicator:indicator_ini},
            grad_op=grad_op,
            fmin_l_bfgs_b_extra_kwargs={'maxfun':15000},
            sess=sess
        )
        print("Final loss:", results_theta['loss'])
        print("Optimization info:", results_theta['info'])
        '''
        #feed_dict = {input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:sa['mask'],theta:results_theta['flows']}
        #feed_dict = {input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:indicator_ini,theta:results_theta['flows']}
        feed_dict = {input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:mask,theta:theta_in}
        '''
         # using lbfgs update spacial transformer
        results = stadv.optimization.lbfgs(
            loss,
            flows,
            # random initial guess for the flow
            flows_x0=np.random.random_sample((seq_len, 2, spec.crop_size,spec.crop_size)),
            feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:ga['mask'],theta:ga['angle']},
            grad_op=grad_op,
            fmin_l_bfgs_b_extra_kwargs={'maxfun':1500},
            sess=sess
        )
        print("Final loss:", results['loss'])
        print("Optimization info:", results['info'])
        feed_dict_new = {input_image: images[0:seq_len], input_label: labels, tau: 0.05,indicator:ga['mask'],theta:ga['angle'],flows:results['flows']}
        '''
        
        
        start_loss = var_loss1
        if ii < 400:
            Test_mode = False
            for cur_iter in range(max_iter):
                start_time = time.time()
                tot_iter += 1
                    
                    
                sess.run(train, feed_dict=feed_dict)
                var_loss,true_prob,var_loss1, var_loss2, var_l12loss,var_pre= sess.run((loss, true_label_prob, loss1, loss2,loss2_l12, pre_label), feed_dict=feed_dict)
                print('iter:', cur_iter, 'total loss:', var_loss, 'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib:', true_prob,'var_l12loss',var_l12loss)
                
                print('time',time.time()-start_time)
                break_condition = False
                '''
                if cur_iter ==1:
                     break_condition = True
                '''
                if var_loss < min_loss:
                    if np.absolute(var_loss-min_loss) < 0.00001:
                        break_condition = True
                        print(last_min)
                        min_loss = var_loss
                        last_min = cur_iter
                
                
                if cur_iter + 1 == max_iter or break_condition:
                    print('iter:', cur_iter,  'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib:', true_prob,'var_l12loss',var_l12loss)
                    var_diff, flows_var, var_probs, noise_norm = sess.run((modifier, flows, probs, norm_frame), feed_dict=feed_dict)
                    
                    #for pp in range(seq_len):
                     #print the map value for each frame
                        #print(noise_norm[0][pp])
                    for i in range(len(indices)):
                        top1 = var_probs[i].argmax()
                        if labels[i] == top1:
                            correct_noi += 1
                    np.save('flow_st_only.npy',flows_var)
                    np.save('modifier_st_only.npy',var_diff)
                    break
                
                    
                
                    
            print('saved modifier paramters.', ii,'spend time',time.time()-start_time)
 
        
       
        
        true_im= sess.run(true_image, feed_dict=feed_dict)
        
     
        for ll in range(len(indices)):
            for kk in range(def_len):
                if kk < seq_len:
                    #if indicator[kk] == 1:
                    
                    attack_image = true_im[ll][kk]
                        #np.reshape(angle_var,(6))
                    
                    attack_img = np.clip(attack_image*255.0+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                        
                    #else:
                        #attack_img = np.clip(images[ll][kk]*255.0+var_diff[0][kk]+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                   
                    diff = np.clip(np.absolute(var_diff[0][kk])*255.0, data_spec.rescale[0],data_spec.rescale[1])
                else:
                   attack_img = np.clip(images[ll][kk]*255.0+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                   diff = np.zeros((spec.crop_size,spec.crop_size,spec.channels))
                im_diff = toimage(arr=diff, cmin=data_spec.rescale[0], cmax=data_spec.rescale[1])
                im = toimage(arr=attack_img, cmin=data_spec.rescale[0], cmax=data_spec.rescale[1])
                new_name = output_name[ll][kk].split('/')
                
                adv_dir = output_file_dir+'/adversarial_100/'
                dif_dir = output_file_dir+'/noise_100/'
                if not os.path.exists(adv_dir):
                   os.mkdir(adv_dir)
                   os.mkdir(dif_dir)

                tmp_dir = adv_dir+new_name[-2]
                tmp1_dir = dif_dir+new_name[-2]
                if not os.path.exists(tmp_dir):
                   os.mkdir(tmp_dir)
                   os.mkdir(tmp1_dir)
               
                new_name = new_name[-1] + '.png'
                im.save(tmp_dir + '/' +new_name)
                im_diff.save(tmp1_dir + '/' +new_name)
  
        
        #print('saved adversarial frames.', ii,file=f)
        
        #print('correct_ori:', correct_ori, 'correct_noi:', correct_noi,'adv_examples',adv,file=f)
        print('correct_ori:', correct_ori, 'correct_noi:', correct_noi,'adv_examples',adv)
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory of output image file.')
    parser.add_argument('--dataset', type=str, required=True,choices=['UCF101','HMDB51'],
                help='dataset to be evaluated.')
    parser.add_argument('--model', type=str, required=True,choices=['GoogleNet','Inception2','i3d_inception','I3V','LSTM','i3d','resnet','tsn','c3d'],
                help='Models to be evaluated.')
    parser.add_argument('--num_images', type=int, default=sys.maxsize,
                        help='Max number of images to be evaluated.')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--num_iter', type=int, default=5,
                        help='Number of iterations to generate attack.')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save .npy file when each save_freq iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.001 * 255,
                        help='Learning rate of each iteration.')
    parser.add_argument('--target', type=str, default=None,
                        help='Target list of dataset.')
    parser.add_argument('--weight_loss2', type=float, default=1.0,
                        help='Weight of distance penalty.')
    parser.add_argument('--not_crop', dest='use_crop', action='store_false',
                        help='Not use crop in image producer.')

    parser.set_defaults(use_crop=True)
    args = parser.parse_args()
    print(args.file_list)
    assert args.num_iter % args.save_freq == 0

    data_spec = models.get_data_spec(model_name=args.model)
    args.learning_rate = args.learning_rate / 255.0 * (data_spec.rescale[1] - data_spec.rescale[0])
    seq_len = 40
    total_len = 40
    batch_size = 1
    targets = None
    if args.target is not None:
        targets = {}
        with open(args.target, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                targets[key] = int(value)
                
    calc_gradients(
        args.file_list,
        args.dataset,
        args.model,
        args.output_dir,
        args.num_iter,
        args.learning_rate,
        targets,
        args.weight_loss2,
        data_spec,
        batch_size,
        total_len,
        seq_len)
    
if __name__ == '__main__':
    main()

