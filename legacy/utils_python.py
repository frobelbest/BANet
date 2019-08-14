import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

def interpolate(imgs_flat,width,height,_x,_y):

    # imgs_shape=imgs.get_shape()
    nbatch    =int(_x.get_shape()[0])
    nchannels =int(imgs_flat.get_shape()[-1])
    # npixels   =int(_x.get_shape()[1])

    height_float=float(height)
    width_float =float(width)


    x=tf.reshape(_x,[-1])
    y=tf.reshape(_y,[-1])

    _x0 = tf.floor(x)
    _y0 = tf.floor(y)

    dx = x-_x0
    dy = y-_y0

    w00 = tf.reshape((1.0-dx)*(1.0-dy),[-1,1,1])
    w01 = tf.reshape(dx*(1.0-dy),[-1,1,1])
    w10 = tf.reshape(((1.0-dx)*dy),[-1,1,1])
    w11 = tf.reshape(dx*dy,[-1,1,1])

    base = tf.reshape(tf.tile(tf.expand_dims(tf.range(nbatch)*height*width,-1),[1,npixels]),[nbatch*npixels])

    x0=tf.cast(_x0,dtype=tf.int32)
    y0=tf.cast(_y0,dtype=tf.int32)
    x1=x0+1
    y1=y0+1
    
    zero = tf.zeros([], dtype='int32')
    x0 = tf.clip_by_value(x0, zero, width-1)
    x1 = tf.clip_by_value(x1, zero, width-1)
    y0 = tf.clip_by_value(y0, zero, height-1)
    y1 = tf.clip_by_value(y1, zero, height-1)

    index00 = base+y0*width+x0
    index01 = base+y0*width+x1
    index10 = base+y1*width+x0
    index11 = base+y1*width+x1

    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)
    output = tf.add_n([tf.matmul(I00,w00),tf.matmul(I01,w01),tf.matmul(I10,w10),tf.matmul(I11,w11)])

    output    = tf.reshape(output,[nbatch,npixels,nchannels])
    cliped_x  = tf.clip_by_value(_x,0.0,width_float-1.0)
    cliped_y  = tf.clip_by_value(_y,0.0,height_float-1.0)
    mask      = tf.expand_dims(tf.to_float(tf.logical_and(tf.equal(_x,cliped_x),tf.equal(_y,cliped_y))),-1)
    return output,mask

def interpolate2d(imgs,_x,_y):
    
    imgs_shape=imgs.get_shape()
    nbatch    =int(imgs_shape[0])
    height    =int(imgs_shape[1])
    width     =int(imgs_shape[2])
    nchannels =int(imgs_shape[3])
    npixels   =int(_x.get_shape()[1])

    height_float=float(height)
    width_float =float(width)


    x=tf.reshape(_x,[-1])
    y=tf.reshape(_y,[-1])

    _x0 = tf.floor(x)
    _y0 = tf.floor(y)

    dx = x-_x0
    dy = y-_y0

    w00 = tf.reshape((1.0-dx)*(1.0-dy),[-1,1,1])
    w01 = tf.reshape(dx*(1.0-dy),[-1,1,1])
    w10 = tf.reshape(((1.0-dx)*dy),[-1,1,1])
    w11 = tf.reshape(dx*dy,[-1,1,1])

    base = tf.reshape(tf.tile(tf.expand_dims(tf.range(nbatch)*height*width,-1),[1,npixels]),[nbatch*npixels])

    x0=tf.cast(_x0,dtype=tf.int32)
    y0=tf.cast(_y0,dtype=tf.int32)
    x1=x0+1
    y1=y0+1
    
    zero = tf.zeros([], dtype='int32')
    x0 = tf.clip_by_value(x0, zero, width-1)
    x1 = tf.clip_by_value(x1, zero, width-1)
    y0 = tf.clip_by_value(y0, zero, height-1)
    y1 = tf.clip_by_value(y1, zero, height-1)

    index00 = base+y0*width+x0
    index01 = base+y0*width+x1
    index10 = base+y1*width+x0
    index11 = base+y1*width+x1

    imgs_flat = tf.reshape(imgs,[nbatch*height*width,nchannels,1])
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)
    output = tf.add_n([tf.matmul(I00,w00),tf.matmul(I01,w01),tf.matmul(I10,w10),tf.matmul(I11,w11)])

    output    = tf.reshape(output,[nbatch,npixels,nchannels])
    cliped_x  = tf.clip_by_value(_x,0.0,width_float-1.0)
    cliped_y  = tf.clip_by_value(_y,0.0,height_float-1.0)
    mask      = tf.expand_dims(tf.to_float(tf.logical_and(tf.equal(_x,cliped_x),tf.equal(_y,cliped_y))),-1)
    return output,mask

def interpolate2d3(imgs,_x,_y):
    
    imgs_shape=imgs.get_shape()
    nbatch    =int(imgs_shape[0])
    height    =int(imgs_shape[1])
    width     =int(imgs_shape[2])
    nchannels =int(imgs_shape[3])
    npixels   =int(_x.get_shape()[1])

    height_float=float(height)
    width_float =float(width)


    x=tf.reshape(_x,[-1])
    y=tf.reshape(_y,[-1])

    _x0 = tf.floor(x)
    _y0 = tf.floor(y)

    dx = x-_x0
    dy = y-_y0

    w00 = tf.reshape((1.0-dx)*(1.0-dy),[-1,1,1])
    w01 = tf.reshape(dx*(1.0-dy),[-1,1,1])
    w10 = tf.reshape(((1.0-dx)*dy),[-1,1,1])
    w11 = tf.reshape(dx*dy,[-1,1,1])

    base = tf.reshape(tf.tile(tf.expand_dims(tf.range(nbatch)*height*width,-1),[1,npixels]),[nbatch*npixels])

    x0=tf.cast(_x0,dtype=tf.int32)
    y0=tf.cast(_y0,dtype=tf.int32)
    x1=x0+1
    y1=y0+1
    
    zero = tf.zeros([], dtype='int32')
    x0 = tf.clip_by_value(x0, zero, width-1)
    x1 = tf.clip_by_value(x1, zero, width-1)
    y0 = tf.clip_by_value(y0, zero, height-1)
    y1 = tf.clip_by_value(y1, zero, height-1)

    index00 = base+y0*width+x0
    index01 = base+y0*width+x1
    index10 = base+y1*width+x0
    index11 = base+y1*width+x1

    imgs_flat = tf.reshape(imgs,[nbatch*height*width,nchannels,1])
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)
    output = tf.add_n([tf.matmul(I00,w00),tf.matmul(I01,w01),tf.matmul(I10,w10),tf.matmul(I11,w11)])

    output    = tf.reshape(output,[nbatch,npixels,nchannels])
    cliped_x  = tf.clip_by_value(_x,0.0,width_float-1.0)
    cliped_y  = tf.clip_by_value(_y,0.0,height_float-1.0)
    mask      = tf.expand_dims(tf.to_float(tf.logical_and(tf.equal(_x,cliped_x),tf.equal(_y,cliped_y))),-1)
    return output,mask,index00,tf.matmul(I00,w00)

def interpolate2d2(imgs,p):

    imgs_shape=imgs.get_shape()
    nbatch    =int(imgs_shape[0])
    height    =int(imgs_shape[1])
    width     =int(imgs_shape[2])
    nchannels =int(imgs_shape[3])
    npixels   =int(p.get_shape()[1])

    height_float=float(height)
    width_float =float(width)


    x=tf.reshape(p[:,:,0],[-1])
    y=tf.reshape(p[:,:,1],[-1])
    # print "x_shape"
    # print x.get_shape()

    _x0 = tf.floor(x)
    _y0 = tf.floor(y)

    dx = x-_x0
    dy = y-_y0

    w00 = tf.reshape((1.0-dx)*(1.0-dy),[-1,1,1])
    w01 = tf.reshape(dx*(1.0-dy),[-1,1,1])
    w10 = tf.reshape(((1.0-dx)*dy),[-1,1,1])
    w11 = tf.reshape(dx*dy,[-1,1,1])

    base = tf.reshape(tf.tile(tf.expand_dims(tf.range(nbatch)*height*width,-1),[1,npixels]),[nbatch*npixels])

    x0=tf.cast(_x0,dtype=tf.int32)
    y0=tf.cast(_y0,dtype=tf.int32)
    x1=x0+1
    y1=y0+1
    
    zero = tf.zeros([], dtype='int32')
    x0 = tf.clip_by_value(x0, zero, width-1)
    x1 = tf.clip_by_value(x1, zero, width-1)
    y0 = tf.clip_by_value(y0, zero, height-1)
    y1 = tf.clip_by_value(y1, zero, height-1)

    index00 = base+y0*width+x0
    index01 = base+y0*width+x1
    index10 = base+y1*width+x0
    index11 = base+y1*width+x1

    imgs_flat = tf.reshape(imgs,[nbatch*height*width,nchannels,1])
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    output = tf.add_n([tf.matmul(I00,w00),tf.matmul(I01,w01),tf.matmul(I10,w10),tf.matmul(I11,w11)])
    output = tf.reshape(output,[nbatch,npixels,nchannels])
    return output

def translations_to_projective_transforms(translations, name=None):
  with ops.name_scope(name, "translations_to_projective_transforms"):
    translation_or_translations = ops.convert_to_tensor(
        translations, name="translations", dtype=dtypes.float32)
    if translation_or_translations.get_shape().ndims is None:
      raise TypeError(
          "translation_or_translations rank must be statically known")
    elif len(translation_or_translations.get_shape()) == 1:
      translations = translation_or_translations[None]
    elif len(translation_or_translations.get_shape()) == 2:
      translations = translation_or_translations
    else:
      raise TypeError("Translations should have rank 1 or 2.")
    num_translations = array_ops.shape(translations)[0]
    return array_ops.concat(
        values=[
            array_ops.ones((num_translations, 1), dtypes.float32),
            array_ops.zeros((num_translations, 1), dtypes.float32),
            -translations[:, 0, None],
            array_ops.zeros((num_translations, 1), dtypes.float32),
            array_ops.ones((num_translations, 1), dtypes.float32),
            -translations[:, 1, None],
            array_ops.zeros((num_translations, 2), dtypes.float32),
        ],
        axis=1)

def translate(images, translations, interpolation="BILINEAR", name=None):
  with ops.name_scope(name,"translate"):
    return tf.contrib.image.transform(
        images,
        translations_to_projective_transforms(translations),
        interpolation=interpolation)