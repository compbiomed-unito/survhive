.PHONY= all test lint 

help:
	@echo arguments: $(.PHONY)

all: lint test

test:
	python3 -m pytest

lint:
	black survwrap/*.py tests/*.py
