import numpy as np
import State
import Vehicle
# Output
# T lines, 2V long.
#
# With ID a vehicle should display
# with ID of ad to be added to cache or -1

# # huge
# V = 31
# T = 25174
# output = open("huge.txt","w")
# file = open("very large.in", "r")

# # large
# V = 152
# T = 4999
# output = open("large.txt","w")
# file = open("large.in", "r")

# # medium
# V = 30
# T = 1192
#output = open("medium.txt","w")
#file = open("medium 2.in", "r")

# # small
# V = 8
# T = 1652
#output = open("small.txt","w")
#file = open("medium 1.in", "r")

# tiny
# V = 3
# T = 99
output = open("tiny.txt","w")
file = open("tiny.in", "r")

line = file.readline()
index = line.find(" ")
N = line[:index]
line = line[index + 1:]
index = line.find(" ")
V = line[:index]
line = line[index + 1:]
index = line.find(" ")
T = line[:index]
line = line[index + 1:]
index = line.find(" ")
M = line[:index]
line = line[index + 1:]
index = line.find(" ")
C = line[:index]
line = line[index + 1:]
index = line.find(" ")
S = line[:index]

P = range(0, int(N))
B = range(0, int(N))
R = range(0, int(N))
X = range(0, int(N))
Y = range(0, int(N))

for i in range(0, int(N)):
    line = file.readline()
    index = line.find(" ")
    P[i] = line[:index]
    line = line[index + 1:]
    index = line.find(" ")
    B[i] = line[:index]
    line = line[index + 1:]
    index = line.find(" ")
    R[i] = line[:index]
    line = line[index + 1:]
    index = line.find(" ")
    X[i] = line[:index]
    line = line[index + 1:]
    index = line.find(" ")
    Y[i] = line[:index]

Vx = [[0 for x in range(0, int(V))] for y in range(0, int(T))]
Vy = [[0 for x in range(0, int(V))] for y in range(0, int(T))]

vehicles = range(0, int(V))
states = [[0 for x in range(0, int(T))] for y in range(0, int(V))]

for i in range(0, int(V)):
    vehicles[i] = Vehicle.Vehicle(M)

for i in range(0, int(T)):
    line = file.readline()
    for j in range(0, int(V)):
        index = line.find(" ")
        Vx[i][j] = line[:index]
        line = line[index + 1:]
        index = line.find(" ")
        Vy[i][j] = line[:index]


state = State.State(N,V, int(T) - 1, M, C, S, P, B, R, X, Y, Vx, Vy, states, vehicles)
test = state.getBestState()

for t in range(int(T)-1, -1, -1):
    for v in range(0, int(V)):
        output.write(str(test[int(v)][int(t)]))
    output.write(("\n"))

output.close()


