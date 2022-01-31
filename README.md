# CAP 4601 Project 2

This project uses natural language processing to create a knowledge database from unstructured text.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/TobiasJacob/cap-4601-project-2)

[Try out this model in Google Colab](https://colab.research.google.com/drive/1ZYmw0MZ83Ce2DdrlxoeBQ6HEuYLJG9jc?usp=sharing)

## Dev setup

This project requires a hell of dependencies. Mostly due to various deep learning frameworks used in the PURE model. There is a docker container provided to deal with those issues.

Install all required packages with

```console
pip install -r requirementsGPU.txt
```

Update the pre-commit hooks

```console
pre-commit install
```

for proper code styling. Check style with

```console
python -m isort .
python -m black .
python -m flake8
```

## Train the models

Run

```console
python -m src.semevalModel.trainEntity
python -m src.semevalModel.trainRelation
```

## Run main

Adjust the paths for the models directory. Run

```console
python -m src.cap_4601_project_2.main
```
