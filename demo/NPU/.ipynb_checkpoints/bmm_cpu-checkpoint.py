import numpy as np
import tensorflow as tf
import time

def main():
    output_data = []
    np.random.seed(10)
    for x in range(5):
        a = np.random.randint(2,size=16)
        b = np.random.randint(2,size=8)
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
        output_data = np.append(output_data,result)
    print(output_data)
    
if __name__ == '__main__':
  main()