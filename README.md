# DRLND-project-1
This repository contains an implementation of project 2 for [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project details

This project implements PPO for solving a Unity environment for continuous control -- making robotic arms follow
rotating waypoints -- on the variant version with 20 agents.

Recording of final implementation:

![Implemented solution](reacher_solved.gif)

Rewards are accumulated over time for each agent while the arm extremity is within the target spheres / within a
fixed range of the target waypoint.  The goal of the agent is to follow the waypoint.

For each agent, the state space has 33 dimensions, and the action space has 4 continuous dimensions.

The task is episodic, and it is considered solved when the agent gets an average score of +30 over 100 consecutive
episodes.

## Getting started

### Dependencies

Dependencies for this project can be setup as per dependencies for the [DRL repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).  The instructions below
walk you through setting up this environment:

1. Create (and activate) a new environment with Python 3.6.
    * Linux or Mac:
    ```
    conda create --name drlnd pythhon=3.6
    source activate drlnd
    ```
    * Windows:
    ```
    conda create --name drlnd pythhon=3.6
    conda activate drlnd
    ```

2. Perform a minimal install of the OpenAI gym, as instructed on [this repository](https://github.com/openai/gym),
or the very helpful instructions at [Open AI Spinning Up](https://spinningup.openai.com/en/latest/user/installation.html).
    * Note that openmpi is not natively supported on Windows 10; I had luck installing instead the [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

3. Clone this repository, and install its dependencies:
    ```
    git clone https://github.com/telmo-correa/DRLND-project-2
    cd DRLND-project-2
    pip install .
    ```
    
    * Note that there seems to be issues installing unityagents on Windows 10 -- conda looks for a required version of
    pytorch that does not seem to be available.  [Commenting out that requirement and installing pytorch separately](https://github.com/udacity/deep-reinforcement-learning/issues/13#issuecomment-475455429)
    worked for me.
 
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the ```drlnd``` environment:
    ```
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

5. Run the Jupyter notebook using the newly created kernel:

![Selecting kernel on Jupyter](drlnd_kernel.png)

### Downloading the Unity environment

Different versions of the Unity environment are required on different operational systems.

Version 1: One (1) Agent

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Version 2: Twenty (20) Agents

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Download the corresponding zip file and unpack it.  You will need to point to the location of the file
within the ```Continuous_Control.ipynb``` and ```Report.ipynb``` notebooks in order to run them.

## Instructions

The problem and the environment are available on the ```Continuous_Control.ipynb``` notebook.

The implementation is provided on the ```Report.ipynb```notebook, along with a discussion of its details.
