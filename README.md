## Setup
Setup Python 3.9:
```
pyenv install 3.9
pyenv local 3.9
poetry env use 3.9
```

## Goal
* Simple game with AI-trained opponent
* Ideally, I could get this running in the browser for super easy sharing

## Todo
* Rename to shadows
* Port to JS

## Bugs
* action IDs may not map properly to agent indices

## Other ideas
* shooting makes you visible
* idea: roadmap to avoid obstacles?

## Learning
* Current approach is to keep this repo and env separate from any RL stuff, and
  just `pip install -e .` it in the RL zoo's venv
* Wrappers seemingly don't work well
  - GrayscaleObservation makes it so I can't use the CnnPolicy
  - ResizeObservation doesn't parse the size properly as a tuple?
* Probably do actually need to put together multiple frames as a memory here,
  since we need to remember where the agent being chased was previously
  - general, the policy needs some memory since there are unobservable states
* May have the policies just always make the agent go forward
* Currently, even with a 1e7 timesteps, the agent just learns a single action
* On the simple environment with variable enemy location, no obstacles, no
  occlusion, size (50, 50), DQN learns well with 1e7 timesteps
  - currently saved with id=6
  - it fails to learn with this same setup if the environment is 3-channel
    rather than grayscale
  - this does **not** work when the local actions are used, but appending the
    agent position and angle to the observation makes it work!
  - shaped rewards (with potential function) also seem to help
  - using enemy speed = 60 and player speed = 50 works well, but upping the
    enemy speed to 75 fails to learn well with 1e7 timesteps (the good run is
    `ppo/Simple-v0_7`
  - using `ppo/Simple-v0_7` as the starting point for with enemy speed = 75
    works well; saved in: `ppo/Simple-v0_10`
  - building once again on `ppo/Simple-v0_10` with only 1e6 timesteps of
    training, the agent is reasonably good with more obstacles added to the
    scene (saved in `ppo/Simple-v0_11`)

* idea: train a model to predict where the other agent is despite occlusions

## Possible other games
### Shooting Game
* same occlusions at the tag game
* you have health and can shoot the other player
* clip size is very limited (same size as max health?), so one needs to reload
  often
* can find health scattered about the map to recover

### Other
* Another idea for a game is to have one applying force to some body while
  bouncing off of other bodies subject to gravity and also shooting to propel
  others away (possibly to propel yourself as well?) and seeking to gather
  treasures

## References
On residual policy learning:
* <https://doi.org/10.1109/ICRA.2019.8794127>
* <https://doi.org/10.48550/arXiv.1812.06298>

