# import the needed libraries
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# werk zonder GPU ?
tf.config.experimental.set_visible_devices([], 'GPU')


width = 100
height = 100
depth = 1
batch_size = 5 #training batch size
testing_data_dir = 'test' #data testing path

# read the input image using OpenCV (you can use another library, e.g., Pillow)
img1 = cv2.imread( testing_data_dir + "/0/boardImg6.jpg",0)
img2 = cv2.imread( testing_data_dir + "/1/rBulletImg4.jpg",0)
img3 = cv2.imread( testing_data_dir + "/2/yBulletImg20.jpg",0)

# convert to ndarray numpy
img1 = np.asarray(img1)
img2 = np.asarray(img2)
img3 = np.asarray(img3)

# load the trained model
model = load_model('model.h5')

# predict the input image using the loaded model
# test1 = model.predict_proba(img1/255).reshape((1,width,height,depth))
pred1 = model.predict_classes((img1/255).reshape((1,width,height,depth)))
pred2 = model.predict_classes((img2/255).reshape((1,width,height,depth)))
pred3 = model.predict_classes((img3/255).reshape((1,width,height,depth)))

# plot the prediction result
print('pred:'+str(pred1[0]))
plt.figure('img1')
plt.imshow(img1,cmap='gray')
plt.title('pred:'+str(pred1[0]), fontsize=22)

plt.figure('img2')
plt.imshow(img2,cmap='gray')
plt.title('pred:'+str(pred2[0]), fontsize=22)

plt.figure('img3')
plt.imshow(img3,cmap='gray')
plt.title('pred:'+str(pred3[0]), fontsize=22)

plt.show()