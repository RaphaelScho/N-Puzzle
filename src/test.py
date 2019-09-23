import math
from profilestats import profile
from collections import deque

@profile
def test():
    print("hi")
    return 7**7

print(test)
print("")

print(round(16/3))



for i in range(1,30):
    print(((1 / math.sqrt(i) - 1) - 0.001)/20)

print()
print(((1 / math.sqrt(9999999999999) - 1) - 0.001)/20)




d1 = deque(maxlen=10)
d2 = deque(maxlen=2)

d2.append((1,2))
d2.append((3,4))
d2.append((5,6))
print (d2)
d1.extend(d2)
print(d1)

d2.append((1,2))
d2.append((3,4))
d1.extend(d2)
print(d1)

d2.append((1,2))
d2.append((3,4))
d1.extend(d2)
print(d1)

d2.append((1,2))
d2.append((3,4))
d1.extend(d2)
print(d1)

d2.append((1,2))
d2.append((3,4))
d1.extend(d2)
print(d1)

d2.append((1,2))
d2.append((3,4))
d1.extend(d2)
print(d1)