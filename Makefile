.PHONY= help all test black lint 

help:
	@echo arguments: $(.PHONY)

all: black ctags lint test docs

test:
	python3 -m pytest tests/*.py

black: 
	black survhive/*.py tests/*.py

docs: black 
	pdoc -o docs survhive

lint: ctags
	ruff survhive/*.py tests/*.py

ctags: 
	ctags-universal -R survhive/ tests/
