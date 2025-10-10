# Face Recognition Duplicate Detection System

This is an mvp-grade face recognition system to detect duplicate faces in a vector database. The app is built with deep learning models and serve using gradio for interactive user experience

## TL;DR quick run
To run and test the application locally, run these steps :

```bash
# 1. Clone repository
git clone https://github.com/yourusername/face-recognition-mvp.git
cd face-recognition-mvp

# Assuming we don't have the images available, then you can run this command

# 2. Download face images into a directory called data/images
make async-dl
# or for sequential process run make seq-dl

# 3. Build vector database, which will be saved in a face_db folder
make build-db

# 4. Start the gradio application
make run
```

Access the application at: http://localhost:7860


## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technical Stack](#technical-stack)
- [Face Recognition Pipeline](#face-recognition-pipeline)
- [Database Architecture](#database-architecture)
- [Gradio Web Application](#gradio-web-application)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Performance Metrics](#performance-metrics)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Architecture Overview

The system implements a complete face recognition pipeline consisting of four main stages:

1. **Preprocessing**: Image loading, format conversion to RGB, and resizing
2. **Face Detection**: Locating and extracting faces using MTCNN or YuNet face recognition model
3. **Embedding Extraction**: Converting faces to 512-dimensional vectors using FaceNet face embedding model
4. **Similarity Search**: Finding duplicates using cosine similarity in ChromaDB

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Module                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Preprocessor │→ │   Detector   │→ │  Embedding   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              ChromaDB Vector Database                       │
│  - Stores 512-D embeddings                                  │
│  - Cosine similarity search                                 │
│  - Metadata and thumbnails                                  │
└─────────────────────────────────────────────────────────────┘
```

## Technical Stack

### Core Libraries

- **Python**: 3.10+
- **PyTorch**: 2.2.2 - Deep learning framework
- **FaceNet-PyTorch**: 2.6.0 - Pre-trained face recognition models
- **ChromaDB**: Latest - Vector database for similarity search
- **Gradio**: 4.7.1 - Web interface framework
- **OpenCV**: 4.8.1.78 - Image processing
- **MTCNN**: 1.0.0 - Face detection

### Supporting Libraries

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Pillow**: Image handling
- **PyYAML**: Configuration management
- **aiohttp**: Asynchronous HTTP client

## Face Recognition Pipeline

### Stage 1: Image Download 

This repo has a csv file containing list of image urls that can be downloaded to your local to build your initial database. 
Two download strategies are provided for building the initial image repository:

**Sequential Download**
- Downloads images one at a time
- More stable, better error handling
- Suitable for smaller datasets (under 1000 images)
- Includes retry logic and rate limiting

**Asynchronous Download**
- Downloads multiple images concurrently
- Significantly faster (10-30x speedup)
- Suitable for large datasets
- Configurable concurrency limits

```bash
# Download images sequentially
make async-dl

# Download images asynchronously (recommended)
make seq-dl
```

### Stage 2: Image Preprocessing

The `ImagePreprocessor` class handles:

1. **Format Conversion**: RGBA to RGB, grayscale to RGB
2. **EXIF Orientation**: Automatic rotation based on EXIF data
3. **Size Optimization**: Resize large images (max 2000px) for performance
4. **Validation**: Check file integrity and format

```python
preprocessor = ImagePreprocessor(max_size=2000)
image = preprocessor.preprocess("path/to/image.jpg")
```

### Stage 3: Face Detection

Two detector options are available:

**MTCNN (Multi-task Cascaded Convolutional Networks)**
- Three-stage cascade architecture (P-Net, R-Net, O-Net)
- Outputs aligned 160x160 face crops
- High accuracy, slower inference (100-200ms per image)
- Default choice for quality-focused applications

**YuNet**
- Single-stage CNN architecture
- Faster inference (20-50ms per image)
- ONNX format for cross-platform compatibility
- Suitable for real-time applications
- Side note : we need to download the YuNet ONNX model file first before using the model

We can confifgure which model to use in our `config.yaml`:

```yaml
face_detection:
  detector_type: "mtcnn"  # or "yunet"
  min_face_size: 50
```

### Stage 4: Embedding Extraction

Uses FaceNet (InceptionResnetV1) to convert face images to embeddings:

- **Input**: 160x160x3 RGB face image
- **Output**: 512-dimensional L2-normalized vector
- **Model Options**:
  - `vggface2`: Trained on VGGFace2 dataset (default, better accuracy)
  - `casia-webface`: Trained on CASIA-WebFace dataset (smaller model)

The embedding space has the property that Euclidean distance corresponds to face similarity. If you have CUDA installed and setup, you can change the device parameter in the FaceNetEmbedding class to 'cuda' to use your GPU for the process

```python
extractor = FaceNetEmbedding(model_name='vggface2', device='cpu')
embedding = extractor.extract(face_tensor)  # Returns 512-D numpy array
```

### Stage 5: Vector Database Storage

We use open source ChromaDB as our vector database for efficient similarity search:

- **Storage**: Persistent disk storage with automatic indexing
- **Distance Metric**: Cosine similarity (1 - cosine distance)
- **Metadata**: Stores person ID, image path, thumbnails
- **Thumbnails**: Base64-encoded 100x100 previews for UI display

```python
db = FaceVectorDB(persist_directory="./face_db")
doc_id = db.add_embedding(embedding, image_path, person_id)
```

### Complete Pipeline Workflow

```
Input Image
    │
    ▼
[Preprocessing]
    │ - Load image
    │ - Fix orientation
    │ - Convert to RGB
    │ - Resize if needed
    ▼
[Face Detection]
    │ - Detect faces
    │ - Select largest face
    │ - Align and crop to 160x160
    ▼
[Embedding Extraction]
    │ - Normalize pixel values
    │ - Forward pass through FaceNet
    │ - L2 normalize output
    ▼
[Database Operations]
    │ - Query for similar embeddings
    │ - Compare distances to threshold
    │ - Return top-K matches
    ▼
Results: Duplicate status + similar faces
```

## Duplicate Search Process

### ChromaDB Schema

Each entry in the database contains:

- **Embedding**: 512-dimensional float32 vector
- **Document**: String reference (image path)
- **ID**: UUID for the entry
- **Metadata**:
  - `person_id`: Name or identifier
  - `image_path`: Original file location
  - `thumbnail`: Base64-encoded preview
  - `processing_time`: Time taken to process
  - `detector`: Which detector was used

### Similarity Search

The system uses cosine similarity for face matching:

```
similarity = 1 - cosine_distance
cosine_distance = 1 - (A · B) / (||A|| × ||B||)
```

Where A and B are L2-normalized embeddings.

**Threshold Configuration**:
- `threshold < 0.4`: Very strict, low false positives
- `threshold = 0.6`: Balanced (default)
- `threshold > 0.8`: Lenient, higher recall

## Gradio Web Application

### Features

1. **Image Upload**: Drag-and-drop or file browser
2. **Duplicate Detection**: Real-time similarity search
3. **Visual Results**: Gallery of top 6 matches with similarity scores
4. **Database Addition**: Optionally add new faces to database
5. **Detailed Results**: JSON view of all match metadata

### Interface Components

**Input Section**:
- Image uploader (accepts JPG, PNG, JPEG)
- Person name text field
- "Add to database" checkbox
- Process button

**Output Section**:
- Status message with duplicate detection results
- Gallery grid showing similar faces with scores
- Expandable JSON panel with detailed match information

### Usage Flow

```
1. User uploads face image
2. Enters person name
3. Clicks "Check for Duplicates"
4. System processes image through pipeline
5. Results displayed:
   - If duplicate found: Shows match info and similarity
   - If not duplicate: Shows closest matches
6. Option to add new face to database
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### System Dependencies

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
```

**macOS**:
```bash
brew install python@3.10
```

**Windows**:
- Install Python from python.org
- Visual C++ Build Tools may be required

### Python Environment

Option 1: Using venv (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Option 2: Using conda
```bash
conda create -n face-recognition python=3.10
conda activate face-recognition
```

### Install Dependencies

```bash

# Install requirement manually:
pip install -r requirements.txt

```

## For New Users

### Clone the repository

```bash
# 1. Clone repository
git clone https://github.com/yourusername/face-recognition-mvp.git
cd face-recognition-mvp

```

### Manual Setup

Assuming we don't have the images available, then run this steps first
```bash
# 1. Download face images 
make async-dl
# or for sequential process run make seq-dl

# 2. Build vector database
make build-db

# 3. Start application
make run
```

Access the application at: http://localhost:7860

## Configuration

Configuration is managed through `config.yaml`:

### Paths Configuration

```yaml
paths:
  data_dir: "./data"              # Training images
  db_dir: "./face_db"             # ChromaDB storage
  logs_dir: "./logs"              # Application logs
  results_dir: "./results"        # Output directory
  yunet_model_path: "./model/..."  # YuNet ONNX model
```

### Face Detection Settings

```yaml
face_detection:
  detector_type: "mtcnn"          # "mtcnn" or "yunet"
  min_face_size: 20               # Minimum face size in pixels
  detection_confidence: 0.9       # Confidence threshold (0.0-1.0)
```

### Embedding Settings

```yaml
embedding:
  embedding_model: "vggface2"     # "vggface2" or "casia-webface"
  embedding_dim: 512              # Fixed at 512 for FaceNet
```

### Duplicate Detection

```yaml
duplicate_detection:
  threshold: 0.6                  # Distance threshold (lower = stricter)
```

### Performance Tuning

```yaml
batch_processing:
  batch_size: 32                  # Images per batch
  num_workers: 4                  # Parallel workers

device:
  type: "cpu"                     # "cpu" or "cuda"
```


## Usage

### Building the Database

```bash
# Clear existing database
make clear-db

# Build from scratch
make build-db
```

The build process:
1. Scans `data/images/` directory
2. Processes each image through the pipeline
3. Extracts person ID from filename (format: `PersonName_ImageID.jpg`)
4. Stores embeddings with metadata and the image thumbnail in ChromaDB

### Running the Application

```bash
# Start Gradio interface
make serve

# Or directly:
python -m gradio src/app.py
```

### Project Structure

```
face-recognition-mvp/
├── src/
│   ├── core/                    # Core business logic
│   │   ├── config.py           # Configuration loader
│   │   ├── detector.py         # Face detection
│   │   ├── embedding.py        # Embedding extraction
│   │   ├── inference.py        # Inference engine
│   │   ├── pipeline.py         # End-to-end pipeline
│   │   └── preprocessing.py    # Image preprocessing
│   ├── db/
│   │   └── vector_db.py        # ChromaDB wrapper
│   ├── downloaders/
│   │   ├── async_download.py   # Async image downloader
│   │   └── sequential_download.py
│   ├── monitoring/
│   │   └── metrics_tracker.py  # Performance metrics
│   ├── utils/
│   │   ├── file_utils.py
│   │   └── validators.py       # Input validation
│   └── app.py                  # Gradio interface
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data
├── config.yaml                 # Main configuration
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container definition
└── Makefile                    # Development commands
```

### Adding New Features

1. **New Detector**: Implement in `src/detector.py`
2. **New Embedding Model**: Modify `src/embedding.py`
3. **Custom Database**: Extend `src/vector_db.py`
4. **UI Changes**: Edit `src/app.py`

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_config.py
│   ├── test_detector.py
│   ├── test_embedding.py
│   ├── test_db.py
│   └── test_preprocessing.py
└── integration/             # Integration tests
    ├── test_end_to_end.py
    └── test_gradio_app.py
```


## Performance Metrics

The system tracks three key metrics:

### 1. End-to-End Latency

Total time from image upload to result display.

**Target**: Under 2 seconds for good UX

```python
# Implementation in src/app.py
start = time.time()
result = process_image(image)
latency_ms = (time.time() - start) * 1000
log_metric("end_to_end_latency", latency_ms)
```

### 2. ChromaDB Query Time

Time spent searching the vector database.

**Target**: Under 500ms for databases with under 10K faces

```python
start = time.time()
results = db.search(embedding, n_results=5)
query_time_ms = (time.time() - start) * 1000
log_metric("chromadb_query_time", query_time_ms)
```

### 3. Face Matching Accuracy

Precision, recall, and F1 score on test set.

**Target**: F1 score above 0.8 for MVP

```python
# Evaluate on test set
tracker = AccuracyTracker()
metrics = tracker.evaluate(detection_function)
# Returns: {precision, recall, f1_score}
```

### Viewing Metrics

```bash
# View latest metrics
make metrics

# Generate summary report
make metrics-summary
```

Metrics are stored in `metrics.jsonl` in JSON Lines format:

```json
{"timestamp": "2025-01-10T12:00:00", "metric": "end_to_end_latency", "latency_ms": 1523.4}
{"timestamp": "2025-01-10T12:00:05", "metric": "chromadb_query_time", "query_time_ms": 234.1}
```

## API Reference

### FaceDuplicateDetector

```python
detector = FaceDuplicateDetector(config_path="./config.yaml")
```

**Methods**:

- `check_duplicate(image_path, n_results=6)`: Check for duplicates
  - Returns: `(is_duplicate, match_info, top_matches)`
  
- `add_to_database(image_path, person_id)`: Add new face
  - Returns: `(success, doc_id)`

### FaceVectorDB

```python
db = FaceVectorDB(persist_directory="./face_db", collection_name="face_embeddings")
```

**Methods**:

- `add_embedding(embedding, image_path, person_id, metadata)`: Add single embedding
- `add_embeddings_batch(embeddings, paths, person_ids, metadatas)`: Batch add
- `search(embedding, n_results)`: Find similar embeddings
- `check_duplicate(embedding, threshold)`: Check if duplicate exists
- `get_count()`: Get total embeddings in database
- `clear()`: Clear all data

## Troubleshooting

### Common Issues

**Issue**: "No face detected"
- **Solution**: Ensure image contains a clear, frontal face
- Check face size is at least 20x20 pixels
- Try adjusting `detection_confidence` in config.yaml

**Issue**: Slow inference
- **Solution**: Use GPU (`device.type: "cuda"` in config)
- Switch to YuNet detector for faster detection
- Reduce image size in preprocessing

**Issue**: High memory usage
- **Solution**: Reduce `batch_size` in config.yaml
- Process fewer images at once
- Use CPU instead of CUDA if VRAM limited

**Issue**: Database not persisting
- **Solution**: Check `db_dir` path is writable
- Verify ChromaDB is properly initialized
- Check for disk space

**Issue**: Import errors
- **Solution**: Ensure all dependencies installed: `make install`
- Check Python version is 3.10+
- Activate virtual environment

### Debug Mode

Enable detailed logging:

```bash
# In .env file
LOG_LEVEL=DEBUG

# Or set environment variable
export LOG_LEVEL=DEBUG
python -m gradio src/app.py
```

### Performance Profiling

```python
# Profile function execution
python -m cProfile -o profile.stats src/pipeline.py

# View results
python -m pstats profile.stats
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Run quality checks: `make pre-commit`
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public functions
- Keep functions under 50 lines when possible

### Testing Requirements

- All new features must include unit tests
- Maintain test coverage above 80%
- Integration tests for end-to-end workflows

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- FaceNet implementation from facenet-pytorch
- MTCNN detector from facenet-pytorch
- YuNet from OpenCV Zoo
- ChromaDB for vector database
- Gradio for web interface

## Citation

If you use this system in your research, please cite:

```bibtex
@software{face_recognition_mvp,
  title = {Face Recognition Duplicate Detection System},
  year = {2025},
  url = {https://github.com/yourusername/face-recognition-mvp}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## Changelog

### Version 1.0.0 (2025-01-10)
- Initial release
- MTCNN and YuNet detector support
- FaceNet embedding extraction
- ChromaDB integration
- Gradio web interface
- Docker containerization
- Comprehensive test suite