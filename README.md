# GamePhysicsSim

This package is a recreational exploration of some concepts and techniques in game creation, control theory and machine learning.

Disclaimer:
It is a work in sporadic progress. Although some effort is being made to keep documentaion up to date and the code moderately clean and free of the most atrocious of bugs, there is currently no guarantee that anything works. Work is still being done to consolidate code from different projects and perfect will not be the enemy of progress here.

The package is currently designed to

- Perform simple physics simulations for moveable objects in games.
- Simulations and visualisation of pod racer example (Inspired by https://www.codingame.com/multiplayer/bot-programming/coders-strike-back). This includes
    - Manual/interactive control mode
    - Automatic movement using PID controller

- Train an AI to move pod between checkpoints, given some physics. This is currently explored in the notebooks.


Spaceships images in Images/ are designed by freepik (www.freepik.com).


----------------------------

Files:
GamePhysicsSim/
    __init__.py
        package init file

    Config.py
        contains a dictionary for settings of e.g. physical quantities (friction etc) that are used across the other submodules

    EvolutionSim.py
        This runs a simulation of a number of pods controlled by a hand tuned PID controller, showing live animations using pygame. It is slow and has no productive purpose at the moment. It can be used as a source for code snippets for running pygame (including calling important rotation animations of objects from Visualise.py) and pid controllers

    NN.py
        Contains functions to load and handle keras models. In particular it also contains the obsolete genetic algorithm functions used in the first iteration of this project. These have since been replaced by the rudimentary GeneticAlgorithm package.

    PhysicalObjects.py
        Contains the RigidBody class.

    PID_run.py
        Runs a simulation where M pods race through a track of N checkpoints, where the steering is governed by a PID regulator for the Torque and a distance based function for the Thrust. If this automatic steering is sufficiently well calibrated you can use the output from one of these runs to train the AI.

    Pod.py
        Contains the Pod class. This class inherits from the RigidBody class and contains all the functions to get that thing to move around. Also functions to steer with PID and NNs.

    RunGA.py
        Newer version of Sim.py and Sim_cont.py, using the GeneticAlgorithm package to keep track of generations and mutations.

        Meant to replace all three of EvolutionSim.py, Sim.py and Sim_cont.py.

    Sim.py
        Runs simulations using old genetic algorithms. Also saves films of each generation. Starts from a new model (i.e. generation 1. in the current nomenclature this makes a difference. Though it would be better if it didn't). The example here starts from a model that has been trained the regular way against a PID.

    Sim_cont.py
        This starts the simulation from a generation later than 0. (This could have been accomplished using the Sim.py submodule, but this lead to less changing code back and forth).

        Both Sim.py and Sim_cont.py are going to be discontinued. However, they contain useful code snippets for evaluating fitness and for plotting animations.

    Utils.py
        utility functions for all submodules. Currently it mainly contains functions to deal with vectors and to calculate angles.

    Visualise.py
        Contains animation and visualisation functions for both pygame and matplotlib based animations, including rotation animations of objects.

        Running Visualise.py opens a manual control window. Good for gauging the current physics settings.

    old/
        Storage for obsolete submodules and some scripts and files. They are saved here temporarily until I know that they contain no code that I wish to reuse.

    profile.txt
        Example of profiling of code. This shows that the most time consuming part of the simulations is the forward propagation through the NNs. I am still not sure this should be the case for such small networks that I use here. I think some more light weight implementation of the NNs might speed things up signifiantly. However, Keras is damned convenient in most other regards.

------------------------------

Q&A:
- Why do we have to submit things like Drag and Friction to pod.move etc, instead of just loading the config file into the podclass?
    - Because we want to be able to adjust these values based on the terrain in a clear way. Maybe not the best solution, but it will do for now.
- 

TO DO:
- [] Clean up significantly in code (refactor from ground up)
  - I'm pretty sure decorators will be a helpful addition to clean things up
  - Some entire submodules are probably going to vanish
- [x] Move entire project into this module
  - Thus, this module won't be a stand alone physics simulation software, with additional code for machine learning for the specific case of pod racers, as was originally intended.
  - This includes moving notebooks and images into this project
- [] Change name
