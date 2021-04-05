# GamePhysicsSim

A package to:
- Perform simple physics simulations for objects in games.
- Visualisation for pod racer example, including
  - Manual/interactive control mode
  - Automatic movement using PID regulator
- Train AI to move pod between checkpoints, given some physics

Inspired by https://www.codingame.com/multiplayer/bot-programming/coders-strike-back

Spaceships images are designed by freepik (www.freepik.com).

### This module 101


TO DO:
- [] Clean up significantly in code (refactor from ground up)
  - I'm pretty sure decorators will be a helpful addition to clean things up
  - Some entire submodules are probably going to vanish
- [x] Move entire project into this module
  - Thus, this module won't be a stand alone physics simulation software, with additional code for machine learning for the specific case of pod racers, as was originally intended.
  - This includes moving notebooks and images into this project
- [] Change name
