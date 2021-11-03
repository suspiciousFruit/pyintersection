import pymodule
#import numpy as np

#arr = np.array([1, 2, 4, 5, 1])
a = [
	[1.1, 1.2, 1.3],
	[2.1, 2.2, 2.3],
	[3.1, 3.2, 3.3]	
]

b = [
	[1.1, 1.3, 1.4]
]

arg = (a, b)
print(pymodule.test(arg))
