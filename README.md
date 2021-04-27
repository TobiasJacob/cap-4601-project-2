# CAP 4601 Project 2

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/TobiasJacob/cap-4601-project-2).

## Dev setup

This project requires a hell of dependencies. Mostly due to various deep learning frameworks used in the LUKE model. Luckly, those dependencies can be managed well with poetry. However, it looks like this model only works under linux.

Install all required packages with

```console
pip install requirementsGPU
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
