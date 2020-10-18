# reinforcement_learning
This repository demonstrates a learning approach to the stabilization of a pendulum. Reinforcement Learning is used to learn a policy for balancing a pendulum in the upright position. Reinforcement Learning hereby refers to Approximate Dynamic Programming (ADP), sampling based versions of Dynamic Programming.
In this example the Q function is learned. The Q function is the function of state-action values. This function assigns each possible action in each possible state a value. After the algorithm has finished learning this Q function the optimal policy is derived by taking the highest valued action in the current state.

# Requirements
- [x] MATLAB 2019a

# Implementation Details
Binary space partition is used to divide the state-action space into smaller areas is the Q-function variance is too high. This efficient datastructure allows to represent the Q-function on the state-action space with variable resolution.
In each area of the binary space tree a Q-function is learned via gradient descent. The Q-function is approximated by a linear combination of sine and cosine functions. These sine and cosine functions form an (orthogonal) function basis. 

# Demonstration 
test.m runs the Q-learning algorithm for the stabilization of a pendulum in the upright position.
The resulting policy is shown below.
Note that the maximum actuation is set such that the pendulum cannot get into upright position in one swing.
<img src="https://github.com/janek-gross/reinforcement_learning/blob/master/pendulum.gif?raw=true" />

## License
https://unlicense.org
