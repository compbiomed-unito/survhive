.PHONY= help all test black lint 

help:
	@echo arguments: $(.PHONY)

all: black lint test

test:
	python3 -m pytest

black:
	black survwrap/*.py tests/*.py

lint:
	flake8 --ignore E501 survwrap/*.py tests/*.py

