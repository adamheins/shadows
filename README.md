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
* Port to JS

## Bugs
* AI agent learns to just move backward?
* action IDs may not map properly to agent indices

## Other ideas
* shooting makes you visible

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

## Possible other games
* Another idea for a game is to have one applying force to some body while
  bouncing off of other bodies subject to gravity and also shooting to propel
  others away (possibly to propel yourself as well?) and seeking to gather
  treasures

## References
On residual policy learning:
* <https://doi.org/10.1109/ICRA.2019.8794127>
* <https://doi.org/10.48550/arXiv.1812.06298>

