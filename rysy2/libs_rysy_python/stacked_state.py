from rysy import *


shape = Shape(4, 4, 3)
frames = 4
state = StackedState(shape, frames)

state.random()
state.set(2,2,1, 1.0)
state.next_frame()
state.next_frame()
state.next_frame()
state._print()


print("program done")
