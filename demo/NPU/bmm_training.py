import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

def representative_dataset():
    for _ in range(100):
       data = np.append(np.random.randint(2,size=16),np.random.randint(256,size=1))
#      data = np.random.rand(1, 25)
       yield [data.astype(np.float32)]

# Generate training dataset by running the original algorithm
training_x = []
training_y = []
training_size = 30000
for x in range(training_size):
    a = np.random.randint(2,size=16)
    b = np.random.randint(2,size=8)
    a = np.reshape(a, (4,4))
    b = np.reshape(b, (4,2))
#    print(a)
#    print(b)
    c = np.matmul(a,b)
    c = c >> 1
    a= np.reshape(a, (16))
    b= np.reshape(b, (8))
    c= np.reshape(c, (8))
    result = 0
    for y in range(8):
        result = (result << 1) + c[y]
    training_x = np.append(training_x,np.append(a,b))
    training_y = np.append(training_y,result)
training_x = np.reshape(training_x,(training_size,24))
#print(training_x)
training_y = np.reshape(training_y,(training_size,1))
#print(training_y)
#print(training_y)

# Generate validation dataset by running the original algorithm
validation_x = []
validation_y = []
for x in range(2000):
    a = np.random.randint(2,size=16)
    b = np.random.randint(2,size=8)
    a = np.reshape(a, (4,4))
    b = np.reshape(b, (4,2))
    c = np.matmul(a,b)
    c = c >> 1
    a= np.reshape(a, (16))
    b= np.reshape(b, (8))
    c= np.reshape(c, (8))
    result = 0
    for y in range(8):
        result = (result << 1) + c[y]
    validation_x = np.append(validation_x,np.append(a,b))
    validation_y = np.append(validation_y,result)
validation_x = np.reshape(validation_x,(2000,24))
validation_y = np.reshape(validation_y,(2000,1))

# Create the model/NN
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(24,)),  # input shape required
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(32, activation=tf.nn.relu),
  tf.keras.layers.Dense(16, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])
#inputs = keras.Input(shape=(24,), name="digits")
#x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
#x = layers.Dense(128, activation="relu", name="dense_2")(x)
#x = layers.Dense(64, activation="relu", name="dense_3")(x)
#x = layers.Dense(32, activation="relu", name="dense_4")(x)
#x = layers.Dense(16, activation="relu", name="dense_5")(x)
#outputs = layers.Dense(1, activation="softmax", name="predictions")(x)

# Compile the model/NN
#model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])

#Fit model on training data
history = model.fit(
    training_x,
    training_y,
    batch_size=32,
    epochs=32,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(validation_x, validation_y),
)
#print(history)
#Save as a TF Model
model.save("bmm_npu.tf")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8 # or tf.uint8
converter.inference_output_type = tf.uint8 # or tf.uint8
converter.experimental_new_quantizer = True # It will enable conversion and quantization of MLIR ops
converter.experimental_new_converter = False
tflite_model = converter.convert()
# Save the model as a TFLite model.
with open('bmm_npu.tflite', 'wb') as f:
  f.write(tflite_model)