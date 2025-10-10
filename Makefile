# Makefile
# MODIFIED: Updated paths for new structure

PYTHON = python3

# Download files async
.PHONY: async-dl
async-dl:
	$(PYTHON) src/downloaders/async_download.py  # CHANGED PATH

# Download files sequentially
.PHONY: seq-dl
seq-dl:
	$(PYTHON) src/downloaders/sequential_download.py  # CHANGED PATH

# Build database (clear first, then build)
.PHONY: build-db
build-db: clear-db
	$(PYTHON) src/core/pipeline.py  # CHANGED PATH

# Clear the database
.PHONY: clear-db
clear-db:
	$(PYTHON) -c "from src.db.vector_db import FaceVectorDB; db = FaceVectorDB('./face_db'); db.clear()"  # UPDATED

# Run the webapp
.PHONY: serve
serve:
	gradio src/app.py

# NEW: Run with main.py
.PHONY: run
run:
	$(PYTHON) src/main.py

# NEW: View metrics
.PHONY: metrics
metrics:
	@if [ -f metrics.jsonl ]; then \
		tail -n 20 metrics.jsonl; \
	else \
		echo "No metrics file found"; \
	fi



