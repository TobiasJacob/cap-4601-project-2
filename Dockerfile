FROM python:3.7

ENV PIP_USER=false

RUN export PIP_USER=false
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
