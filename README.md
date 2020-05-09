# Deep RL and Tennis

### Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Installing dependencies

The recommended way of using this repository is through Anaconda.

Set up your world environment:
```
conda create --name tennis python=3.6
```

and activate it:

```
source activate tennis
```
Install dependencies:

```
cd python/
pip install .
```

Since this repository uses jupyter notebook, install the corresponding banana-navigation kernel:

```
python -m ipykernel install --user --name tennis --display-name "tennis"
```
Finally, in jupyter notebook, before running the code, make sure that the appropriate kernel is selected.

## How to run it

Run `deep-tennis`.ipynb notebooks.

(notebook `tennis-separate-actor-critic.ipynb` has a slightly different implementation).

