# CAP 4601 Project 2

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/TobiasJacob/cap-4601-project-2).

## Dev setup

This project requires a hell of dependencies. Mostly due to various deep learning frameworks used in the LUKE model. Luckly, those dependencies can be managed well with poetry. However, it looks like this model only works under linux.

[Install poetry](https://github.com/python-poetry/poetry) with

```console
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
poetry config virtualenvs.in-project true
```

Install all required packages with

```console
poetry install
```

add packages with

```console
poetry add pendulum@latest
```

Update the pre-commit hooks

```console
poetry run pre-commit install
```

for proper code styling. Check style with

```console
poetry run flake8
```

.Check typing with

```console
poetry run mypy .
```

Test the code with pytest

```console
poetry run pytest
```

and get coverage results

```console
poetry run coverage run -m pytest
poetry run coverage report -m
```
