from train_script import input_shape, num_classes, backbone, get_uncompiled_model
from PIL import Image
import cv2
import numpy as np

height, width, _ = input_shape


model = get_uncompiled_model(input_shape, num_classes, backbone)
model.load_weights('/mnt/mydata/dataset/Playment_top_5_dataset/deeplab_top_5_classes_focal_loss_15.h5', by_name=True)
print(model.summary())



image_path = '/mnt/mydata/dataset/Playment_top_5_dataset/test_images/bdd_80c62ee8-96e3f3bf.jpg'

image = np.array(Image.open(image_path))
resized_image = cv2.resize(image, (width, height))
cv2.imwrite('image.jpg', resized_image)

image = image.astype(np.float32)/255


prediction = model.predict(np.expand_dims(resized_image, axis=0))[0]
print(np.unique(prediction))
#threshold here
#prediction = prediction>0.3
prediction = prediction.astype(np.uint8)
print(prediction.shape, prediction.dtype, prediction.max(), prediction.min(), np.unique(prediction).shape)
#labelmap_flat = np.argmax(prediction, axis=-1)
#print(labelmap_flat.shape, labelmap_flat.dtype, labelmap_flat.max(), labelmap_flat.min())

#prediction = prediction.reshape((height, width, -1))

for i in range(num_classes):
	mask = prediction[:,:,i]
	#mask = mask_flat.reshape((height, width))
	#mask = mask>0.5
	print(mask.shape, np.unique(mask), mask.max(), mask.min())
	mask = mask.astype(np.uint8)
	cv2.imwrite('{}.png'.format(i), mask*255)


