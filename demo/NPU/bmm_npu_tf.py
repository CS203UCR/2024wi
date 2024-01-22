import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import time

def main():
    model = tf.saved_model.load('bmm_npu.tf')
# Run inference
    print('----INFERENCE TIME----')
#    print('Note: The first inference on Edge TPU is slow because it includes',
#            'loading the model into Edge TPU memory.')
    output_data = []
    np.random.seed(10)
    for x in range(5):
        a = np.random.randint(2,size=16)
        b = np.random.randint(2,size=8)
        input_data = np.append(a,b)
        input_data = np.float32(np.reshape(input_data,(1,24)))
        start = time.perf_counter()
        result = model(input_data)
        inference_time = time.perf_counter() - start
        print('%.2f us' % (inference_time*1000000))
        output_data = np.append(output_data,np.round(result))
    print(output_data)
    
if __name__ == '__main__':
  main()