
# Simple Makefile with clear-db command

PYTHON = python3

# Download files async
.PHONY: async-dl
async-dl:
	$(PYTHON) src/image_downloader/async_download.py

# Download files async
.PHONY: seq-dl
seq-dl:
	$(PYTHON) src/image_downloader/sequential_download.py


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

