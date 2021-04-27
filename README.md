# PyDeepRecSys

This project is a WIP. Please don't pay attention to it.

# Python Version

This project, due to its dependencies, is meant to be used with Python `3.6`.

# Installing Dependencies

## Requirements

This poetry uses `poetry` for dependency management. It's also a good idea to create a new virtual environment for this project.

```bash
python3.6 -m venv venv
source ./venv/bin/activate
pip install pip==21.0.1 setuptools==56.0.0 wheel==0.36.2
pip install poetry==1.1.6
poetry install
```

## Preparing ML Fairness Gym

```bash
# prepare submodules
git submodule init
git submodule update --remote
# install ml fairness using custom setup script
python mlfairnessgym.setup.py install
# download movielens data
python -m mlfairnessgym.environments.recommenders.download_movielens
```

## Running Tests

This project uses `pytest` and `coverage`.

```bash
# from project root
poetry run coverage run -m pytest
poetry run coverage html -d html/coverage
```

## Generate Docs

```bash
# from project root
poetry run pdoc --html pydeeprecsys
```



## Short term TODO list

- [ ] Find out a proper learning gym env
- [ ] Make sure DDQN works properly
- [ ] ...


# Long term TODO list
- [ ] CI
- [ ] Precommit hooks (black, flake8, mypy, ...)
