.PHONY= help all test black lint 

help:
	@echo arguments: $(.PHONY)

all: black ctags lint test

test:
	python3 -m pytest tests/*.py

black: 
	black survhive/*.py tests/*.py

lint: ctags
	ruff survhive/*.py tests/*.py

ctags: 
	ctags-universal -R survhive/ tests/
