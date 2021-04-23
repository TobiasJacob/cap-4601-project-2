FROM python:3.7

ENV PIP_USER=false

RUN export PIP_USER=false
RUN pip install poetry
RUN poetry config virtualenvs.in-project true

RUN groupadd -g 1000 -o tobi
RUN useradd -m -u 1000 -g 1000 -o -s /bin/bash tobi
USER tobi
