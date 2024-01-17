import numpy as np

i = np.int16(1)
#dt = np.dtype('i')
#print(dt)
#i = 1
while i > 0:
  i *= 2
#  if i % 1000000 == 0:
#    print (i)

print("We're done with integer loop")

i = 1.0
while i > 0:
  i *= 2.0

print("We're done with float loop")
