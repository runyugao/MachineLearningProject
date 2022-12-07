import os
import argparse
import tensorflow as tf
import numpy as np
import PIL
import time
import matplotlib 
import matplotlib.pyplot as plt
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
matplotlib.rcParams['figure.figsize'] = (12, 12)
matplotlib.rcParams['axes.grid'] = False

#Helper Function##################################################################################################################################################################
def TensorToImage(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def LoadImage(PathToImage):
  img = tf.io.read_file(PathToImage)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  ImageShape = tf.cast(tf.shape(img)[:-1], tf.float32)
  img = tf.image.resize(img, tf.cast(ImageShape * (512 / max(ImageShape)), tf.int32))
  resImage = img[tf.newaxis, :]
  return resImage

def SaveImage(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  tf.keras.utils.save_img(title,image)

#Function about VGG##################################################################################################################################################################
def VGGLayers(LayerName):
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = []
  for name in LayerName:
    outputs.append(vgg.get_layer(name).output)
  return tf.keras.Model([vgg.input], outputs)

def GramMatrix(InputTensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', InputTensor, InputTensor)
  input_shape = tf.shape(InputTensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

#Style Content Model##################################################################################################################################################################
class StyledContentModel(tf.keras.models.Model):
  def __init__(self, StyleLayers, ContentLayers):
    super(StyledContentModel, self).__init__()
    self.vgg = VGGLayers(StyleLayers + ContentLayers)
    self.StyleLayers = StyleLayers
    self.ContentLayers = ContentLayers
    self.NumStyleLayer = len(StyleLayers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    outputs = self.vgg(tf.keras.applications.vgg19.preprocess_input(inputs))
    StyleOutputs, ContentOutputs = (outputs[:self.NumStyleLayer], outputs[self.NumStyleLayer:])
    StyleOutputs = [GramMatrix(style_output) for style_output in StyleOutputs]
    ContentDict = {content_name: value for content_name, value in zip(self.ContentLayers, ContentOutputs)}
    StyleDict = {style_name: value for style_name, value in zip(self.StyleLayers, StyleOutputs)}
    return {'content': ContentDict, 'style': StyleDict}

#Loss Function##################################################################################################################################################################
def StyleContentLoss(outputs):
    StyleOutputs = outputs['style']
    ContentOutputs = outputs['content']
    StyleLoss = tf.add_n([tf.reduce_mean((StyleOutputs[name]-StyleTargets[name])**2) for name in StyleOutputs.keys()])
    StyleLoss *= 1e-2 / 5
    ContentLoss = tf.add_n([tf.reduce_mean((ContentOutputs[name]-ContentTargets[name])**2) for name in ContentOutputs.keys()])
    ContentLoss *= 1e4
    loss = StyleLoss + ContentLoss
    return loss
    
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = StyleContentLoss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


#Main Function##################################################################################################################################################################
#Take input arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--content_path', default = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'), 
                help='please input the path to content image')
ap.add_argument('-s', '--style_path', default=tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'),
                help='please input the path to style image')
ap.add_argument('-e', '--epoch', default=10, type=int,
                help='please set the epochs')
ap.add_argument('-t', '--steps_per_epoch', default=100, type=int,
                help='please set the steps per epoch')
args = ap.parse_args()

#Load and show the content image and the style image
ContentImage = LoadImage(args.content_path)
StyleImage = LoadImage(args.style_path)
#SaveImage(ContentImage, 'ContentImage.png')
#SaveImage(StyleImage, 'StyleImage.png')

#Preprocess and resize the image
x = tf.keras.applications.vgg19.preprocess_input(ContentImage*255)
x = tf.image.resize(x, (224, 224))
#Load VGG19 CNN model and print its structure
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
for l in vgg.layers:
  print(l.name)
ContentLayers = ['block5_conv2'] 
StyleLayers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

#Using 
StyleExtractor = VGGLayers(StyleLayers)
StyleOutputs = StyleExtractor(StyleImage*255)
extractor = StyledContentModel(StyleLayers, ContentLayers)
StyleTargets = extractor(StyleImage)['style']
ContentTargets = extractor(ContentImage)['content']
Image = tf.Variable(ContentImage)

#Train and record the time
start = time.time()
epochs = args.epoch
steps_per_epoch = args.steps_per_epoch

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(Image)
    print("*", end='', flush=True)
  print("Train step: {}".format(step))
SaveImage(TensorToImage(Image), './training/StyledContent.png')
end = time.time()
print("Total time: {}".format(end-start))

