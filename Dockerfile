FROM python:3.7

RUN pip install poetry
RUN poetry config virtualenvs.in-project true
