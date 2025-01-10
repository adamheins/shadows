# Shadows

Small 2D games with computer-controlled opponents trained using reinforcement
learning.

You can play a demo in the browser [here](https://adamheins.com/projects/shadows/web/).

The goal of this project is to explore simple but fun game mechanics, machine
learning for games, and deploying machine learning in the browser.

The main library is written in Python. Currently, a tag game and associated
gymnasium environment for reinforcement learning is implemented. Training is
done with [Stable Baselines 3](https://stable-baselines3.readthedocs.io/).

## Setup

Note that Python 3.9 or greater is required. Installation is done with
[poetry](https://python-poetry.org/):
```
git clone git@github.com:adamheins/shadows
cd shadows
poetry install
```

## Play

Play the game:
```
poetry run scripts/games/tag.py
```

If you have trained some models (more on this below), you can supply models for
the computer-controlled agent to use when "it" and "not it" (recall this is a
game of tag), using the options:
```
poetry run scripts/games/tag.py --it-model <path_to_it_model.zip> \
                                --not-it-model <path_to_not_it_model.zip>
```
If one or both are missing, then a default hand-crafted policy for the agent is
used.

## Train

As mentioned above, models for when the computer agent is "it" and "not it" are
trained separately. This library provides the corresponding gymnasium
environments `TagIt-v0` and `TagNotIt-v0`, respectively. To train, use
`scripts/learn/train.py`. For example, to train a soft actor-critic model on
the `TagIt-v0` env for 1,000,000 timesteps and log the results:
```
poetry run scripts/learn/train.py TagIt-v0 -L logs --algo sac -n 1_000_000
```

## Browser

A limited Javascript-version of the library is also implemented [here](./web).
Models trained using Python can be exported to the portable
[ONNX](https://onnx.ai/) format using `scripts/learn/export_onnx_model.py`.

## License

MIT
