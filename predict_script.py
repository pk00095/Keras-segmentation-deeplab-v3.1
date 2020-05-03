from train_script import input_shape, num_classes, backbone, get_uncompiled_model
from PIL import Image
import cv2
import numpy as np

model = get_uncompiled_model(input_shape, num_classes, backbone)
print(model.summary())


'''

height, width, _ = input_shape

image_path = ''

image = Image.open(image_path)
image = np.array(image, dtype=np.float32)

resized_image = cv2.resize(image, (width, height))

prediction = model.predict(np.expand_dims(resized_image, axis=0))
'''