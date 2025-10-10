
# Simple Makefile with clear-db command

PYTHON = python3

# Download files async
.PHONY: async-dl
async-dl:
	$(PYTHON) src/downloader/async_download.py

# Download files async
.PHONY: seq-dl
seq-dl:
	$(PYTHON) src/downloader/sequential_download.py

# Build database (clear first, then build)
.PHONY: build-db
build-db: clear-db
	$(PYTHON) src/pipeline.py 

# Clear the database
.PHONY: clear-db
clear-db:
	$(PYTHON) src/utils.py

# Run the webapp
.PHONY: serve
serve:
	gradio src/app.py




