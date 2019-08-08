import math
from profilestats import profile

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