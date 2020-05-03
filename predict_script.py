from train_script import input_shape, num_classes, backbone, get_uncompiled_model
from PIL import Image
import cv2
import numpy as np

height, width, _ = input_shape

model = get_uncompiled_model(input_shape, num_classes, backbone)
model.load_weights('/mnt/mydata/dataset/Playment_top_5_dataset/checkpoints/deeplab_top_5_classes_30.h5', by_name=True)
print(model.summary())



image_path = '/mnt/mydata/dataset/Playment_top_5_dataset/test_images/bdd_7d15b18b-1e0d6e3f.jpg'

image = np.array(Image.open(image_path))
resized_image = cv2.resize(image, (width, height))

image = image.astype(np.float32)


prediction = model.predict(np.expand_dims(resized_image, axis=0))[0]
#threshold here
prediction = prediction.reshape((height, width, -1))
perdiction = np.argmax(prediction, axis=-1)

print(prediction.shape, prediction.dtype, prediction.max(), prediction.min())