
# Simple Makefile with clear-db command

PYTHON = python3

# Clear the database
.PHONY: clear-db
clear-db:
	$(PYTHON) src/utils.py

# Build database (clear first, then build)
.PHONY: build-db
build-db: clear-db
	$(PYTHON) src/pipeline.py 

# Run pipeline
.PHONY: run
run:
	$(PYTHON) src/pipeline.py