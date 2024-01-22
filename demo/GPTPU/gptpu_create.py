#https://www.tensorflow.org/lite/models/convert/convert_models#convert_concrete_functions_
import numpy as np
import tensorflow as tf
import time
class BMM(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(24), dtype=tf.float32)])
    def __call__(self, input):
        a = input[0:16]
        b = input[16:24]
        a = tf.reshape(a, [4,4])
        b = tf.reshape(b, [4,2])
        c = tf.linalg.matmul(a,b)
        divisor = 2.0
        c = tf.math.divide_no_nan(c,divisor)
        return c



def main():
    np.random.seed(10)
    for x in range(5):
        a = np.random.randint(2,size=16)
        b = np.random.randint(2,size=8)
        input = np.append(a,b)
        a = np.reshape(a, (4,4))
        b = np.reshape(b, (4,2))
        start = time.perf_counter()
        c = np.matmul(a,b)
        c = c >> 1
        inference_time = time.perf_counter() - start
        print('%.2f us' % (inference_time*1000000))
        # Debugging
        result = 0
        c = np.reshape(c, (8))
        for y in range(8):
            result = (result << 1) + c[y]
        print(result)
    input = np.random.randint(2,size=24)
    model = BMM()
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],model)
    tflite_model = converter.convert()

    # Save the model.
    with open('bmm_gptpu.tflite', 'wb') as f:
      f.write(tflite_model)
if __name__ == '__main__':
  main()