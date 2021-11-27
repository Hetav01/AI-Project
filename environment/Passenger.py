import random

# Passengers are the objects that are to be transported to 
# Different Floors
# The number and destination of the passengers is to be generated randomly
# for reduced inputs

class Passenger(object):
    def __init__(self,now_floor : int,max_floor : int):
        self.now_floor = now_floor
        self.dest = random.choice(list(range(now_floor)) + list(range(now_floor+1,max_floor)))

    # Getting destination
    def get_dest(self) -> int:
        return self.dest    

