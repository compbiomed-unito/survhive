.PHONY= help all test black lint 

help:
	@echo arguments: $(.PHONY)

all: black ctags lint test

test:
	python3 -m pytest

black:
	black survwrap/*.py tests/*.py

lint: black ctags
	ruff survwrap/*.py tests/*.py

ctags: black
	ctags-universal -R survwrap/ tests/
