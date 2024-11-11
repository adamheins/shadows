== Setup ==
Setup Python 3.9:
```
pyenv install 3.9
pyenv local 3.9
poetry env use 3.9
```

== Goal ==
* Simple game with AI-trained opponent
* Ideally, I could get this running in the browser for super easy sharing

== Todo ==
* Port to JS
* Build gym interface
* Idea: instead of knowing exactly where the player is, use the policy itself
  to roll out the player's predicted actions until they are actually observed
  again

== Other ideas ==
* Idea: shoot makes you visible

== Learning ==
* Current approach is to keep this repo and env separate from any RL stuff, and
  just `pip install -e .` it in the RL zoo's venv
* Do I need to specify wrappers for the env?

== Possible other games ==
* Another idea for a game is to have one applying force to some body while
  bouncing off of other bodies subject to gravity and also shooting to propel
  others away (possibly to propel yourself as well?) and seeking to gather
  treasures

== References ==
On residual policy learning:
* <https://doi.org/10.1109/ICRA.2019.8794127>
* <https://doi.org/10.48550/arXiv.1812.06298>

