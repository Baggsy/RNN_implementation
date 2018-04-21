import Vehicle

class State:
    def __init__(self, N, V, T, M, C, S, P, B, R, X, Y, Vx, Vy, states, vehicles):
        self.N = int(N)
        self.V = int(V)
        self.T = int(T)
        self.M = int(M)
        self.C = int(C)
        self.S = int(S)
        self.P = P
        self.B = B
        self.R = R
        self.X = X
        self.Y = Y
        self.Vx = Vx
        self.Vy = Vy
        
        self.states = states
        
        self.vehicles = vehicles

    def getBestState(self):
        if self.T < 0:
            return self.states
        for i in range(0, self.V):
            ad = self.getBestAd(i)
            remove = self.vehicles[i].addToCache(ad)
            self.states[i][self.T] = str(ad) + " " + str(remove) + " "
            
        state = State(self.N, self.V, self.T - 1, self.M, self.C, self.S, self.P, self.B, self.R, self.X, self.Y, self.Vx, self.Vy, self.states, self.vehicles)
        return state.getBestState()

    def getBestAd(self, v):
        profit = [0 for _ in xrange(self.N)]
        bool = False
        for n in range(0, self.N):
            if int(self.B[n]) - int(self.P[n]) >= 0:
                if ((int(self.X[n]) - int(self.Vx[self.T][v])) ^ 2 + (int(self.Y[n]) - int(self.Vy[self.T][v])) ^ 2 < int(self.R[n]) ^ 2):
                    #if int(self.Vx[v][self.T]):
                    profit[n] = self.P[n]
                    
                    bool = True

        chosen_ad = profit.index(max(profit))
        if bool:
            self.B[chosen_ad] = int(self.B[chosen_ad]) - int(self.P[chosen_ad])

        return chosen_ad
