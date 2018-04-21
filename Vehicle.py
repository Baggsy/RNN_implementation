class Vehicle:
    def __init__(self, M):
        self.M = int(M)-1
        self.cached = range(0, int(M))
        self.addedIndex = 0
        for i in range(0,int(M)):
            self.cached[i] = -1

    # Removes oldest from cache
    def addToCache(self, i):
        for j in range(0,int(self.M)):
            if i == self.cached[j]:
                return -1
                       
        # if cache is not full
        toBeRemoved = self.cached[self.addedIndex]
        self.cached[self.addedIndex] = i

        # increment oldest counter
        self.addedIndex = self.addedIndex + 1
        # if oldest counter out of bounds, reset
        if self.addedIndex == self.M:
            self.addedIndex = 0

        # return the object from cache to be removed
        return toBeRemoved
            
