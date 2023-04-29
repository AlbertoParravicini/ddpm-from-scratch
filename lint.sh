#!/bin/sh
isort .
black .
flake8 .
mypy .