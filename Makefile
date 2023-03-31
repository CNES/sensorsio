all: yapf fix-imports pylint mypy pytest
yapf:
	yapf -i -r .
fix-imports:
	ruff check . --select F401 --fix
	isort .
pylint:
	pylint --exit-zero src/sensorsio/*.py tests/*.py
mypy:
	mypy src/sensorsio/*.py tests/*.py
pytest:
	pytest --cov


