import sys
sys.path.append("../libs_rysy_python")

from rysy import *

state_shape   = Shape(48, 48, 12)
actions_count = 5
dqn = DQNCuriosity(state_shape, actions_count, "dqn_curiosity/")


dqn._print()

#dqn.save("dqn_curiosity/")
