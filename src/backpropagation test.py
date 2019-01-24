import math

a = 0.1 # learning rate
x = 1 # which input to feed to network
err_thresh = 0.001 # how much should the error be minimized

# basically learn to to abs(x)
i1_1 = -1.3
i2_1 = 2
exp_o1_1 = 1.3
exp_02_1 = 2

i1_2 = 1
i2_2 = -0.5
exp_o1_2 = 1
exp_02_2 = 0.5


w1 = 0.1
w2 = 0.8
w3 = 0.22
w4 = 0.78

w5 = 0.9
w6 = 0.7
w7 = 0.6
w8 = 0.2

'''
[i1] - w1 - [h1] - w5 - [o1]
    \ w2    /   \ w6    /
      \   /       \   /
        X           X
      /   \       /   \
    / w3    \   / w7    \
[i2] - w4 - [h2] - w8 - [o2]
'''

def sigmoid(x):
    return 1/(1+math.e**(-x))

def calc(in1, in2, weight1, weight2):
    return in1 * weight1 + in2 * weight2

def calc_h(x):
    if x == 1:
        return sigmoid(calc(i1_1, i2_1, w1, w2)),\
            sigmoid(calc(i1_1, i2_1, w3, w4))
    if x == 2:
        return sigmoid(calc(i1_2, i2_2, w1, w2)), \
               sigmoid(calc(i1_2, i2_2, w3, w4))

h1, h2 = calc_h(1)

def calc_o():
    return calc(h1, h2, w5, w6),\
           calc(h1, h2, w7, w8)

o1, o2 = calc_o()

def calc_delta(x):
    if x == 1:
        return o1 - exp_o1_1, o2 - exp_02_1
    if x == 2:
        return o1 - exp_o1_2, o2 - exp_02_2

d1, d2 = calc_delta(1)

print("h1: " + str(h1))
print("h2: " + str(h2))
print("o1: " + str(o1))
print("o2: " + str(o2))
print("d1: " + str(d1))
print("d2: " + str(d2))
print("\n")

def update_w(x):
    # new w5 is old w5 - learning rate * outputnode1-error * output of connected previous node
    new_w5 = w5 - a*d1*h1
    new_w6 = w6 - a*d1*h2
    new_w7 = w7 - a*d2*h1
    new_w8 = w8 - a*d2*h2

    if x == 1:
        # new w1 is old w1 - a * output of connected input node * (o1-error*w5) * (o2-error*w7)
        # so: learn * output of input node * Product of (receiving node error * weight of connection to that node) for each receiving node
        new_w1 = w1 - a * i1_1 * (d1 * w5) * (d2 * w7)
        new_w2 = w2 - a * i2_1 * (d1 * w5) * (d2 * w7)
        new_w3 = w3 - a * i1_1 * (d1 * w6) * (d2 * w8)
        new_w4 = w4 - a * i2_1 * (d1 * w6) * (d2 * w8)

    if x == 2:
        new_w1 = w1 - a * i1_2 * (d1 * w5) * (d2 * w7)
        new_w2 = w2 - a * i2_2 * (d1 * w5) * (d2 * w7)
        new_w3 = w3 - a * i1_2 * (d1 * w6) * (d2 * w8)
        new_w4 = w4 - a * i2_2 * (d1 * w6) * (d2 * w8)

    return new_w5, new_w6, new_w7, new_w8, new_w1, new_w2, new_w3, new_w4

while (abs(d1) > err_thresh) | (abs(d2) > err_thresh):
    if x == 1:
        x = 2
    else:
        x = 1

    # calc new weights
    w5, w6, w7, w8, w1, w2, w3, w4 = update_w(x)

    print("new weights")
    print("w1: " + str(w1))
    print("w2: " + str(w2))
    print("w3: " + str(w3))
    print("w4: " + str(w4))
    print("w5: " + str(w5))
    print("w6: " + str(w6))
    print("w7: " + str(w7))
    print("w8: " + str(w8))

    # calc new h
    h1, h2 = calc_h(x)
    print("new h")
    print("h1: " + str(h1))
    print("h2: " + str(h2))

    # calc new output
    o1, o2 = calc_o()
    print("new out")
    print("o1: " + str(o1))
    print("o2: " + str(o2))

    # calc new delta
    d1, d2 = calc_delta(x)
    print("new delta")
    print("d1: " + str(d1))
    print("d2: " + str(d2))
    print("\n")