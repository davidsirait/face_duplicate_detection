import gradio as gr
from PIL import Image, ImageDraw
import io
import base64
from typing import Tuple, List
import json
import sys
import time
sys.path.append("./src")

from core.inference import FaceDuplicateDetector
from core.config import Config
from utils.validators import validate_image_file, validate_person_name
from monitoring.metrics_tracker import MetricsTracker

# Load configuration
config = Config("./config.yaml")

# Initialize metrics tracker
metrics_tracker = MetricsTracker(
    metrics_file="metrics.jsonl"
)

class FaceRecognitionApp:
    def __init__(self, config_path = "./config.yaml"):
        """Initialize the Face Recognition App"""
        self.predictor = FaceDuplicateDetector(config_path=config_path)
        self.metrics_tracker = metrics_tracker
        
    def process_uploaded_image(
        self, 
        image: str,  
        person_name: str, 
        add_to_db: bool
    ) -> Tuple[str, List[Tuple[Image.Image, str]], str]:
        """
        Process uploaded image and check for duplicates
        WITH INPUT VALIDATION AND METRICS TRACKING
        """
        # Track end-to-end latency
        with self.metrics_tracker.track_latency("end_to_end"):
            try:
                # INPUT VALIDATION (HIGH PRIORITY FIX #1)
                is_valid, error_msg = validate_image_file(image)
                if not is_valid:
                    return f"Invalid image: {error_msg}", [], "{}"
                
                is_valid, error_msg = validate_person_name(person_name)
                if not is_valid:
                    return f"Invalid name: {error_msg}", [], "{}"
                
                # Track ChromaDB query time separately
                start_query = time.time()
                
                # check for duplicates
                is_duplicate, match_info, top_matches = self.predictor.check_duplicate(image)
                
                query_time_ms = (time.time() - start_query) * 1000
                
                # Log ChromaDB query metric
                self.metrics_tracker.log_metric(
                    "chromadb_query_time",
                    round(query_time_ms, 2),
                    {
                        "n_results": len(top_matches.get('ids', [[]])[0]),
                        "status": "success"
                    }
                )

                # process top_matches results
                gallery_images_with_captions = self.process_search_results_with_scores(top_matches)
                
                # Prepare status message
                if is_duplicate:
                    similarity_score = max(0.0, 1 - match_info['distance']) # prevent negative similiarity score
                    status_msg = f"<h1>**Duplicate Found!**</h1>\n"
                    status_msg += f"Similarity Score: {similarity_score:.2%}<br />"
                    status_msg += f"Matched Person: {match_info.get('metadata', {}).get('person_id', 'Unknown')}"
                else:
                    status_msg = "<h1>**No duplicate found**</h1>"
                
                # Add to database if requested and not duplicate
                if add_to_db and not is_duplicate:
                    success, doc_id = self.predictor.add_to_database(image, person_id = person_name)
                    if success:
                        status_msg += f"\n\n**Added to database** with ID: {doc_id[:8]}..."
                    else:
                        status_msg += "\n\nNot added to database, error occured"
                elif add_to_db and is_duplicate:
                    status_msg += "\n\nNot added to database (duplicate detected)"
                
                # Prepare detailed JSON results
                metadatas = top_matches.get('metadatas')
                distances = top_matches.get('distances')

                details = {
                    "person_name": person_name,
                    "is_duplicate": is_duplicate,
                    "num_matches": len(metadatas[0]),
                    "matches": [
                        {
                            "rank": i + 1,
                            "similarity_score": float(1 - distances[0][i]),
                            "distance": float(distances[0][i]),
                            "person_id": metadatas[0][i].get('person_id', 'Unknown') if i < len(metadatas[0]) else 'Unknown'
                        }
                        for i in range(min(6, len(metadatas[0])))
                    ] if distances else [],
                }
                return status_msg, gallery_images_with_captions, json.dumps(details, indent=2)
                
            except Exception as e:
                return f"❌ Error: {str(e)}", [], "{}"
    
    def process_search_results_with_scores(self, search_results: dict) -> List[Tuple[Image.Image, str]]:
        """
        Convert search results to list of (PIL Image, caption) tuples
        Caption includes similarity score and person ID
        """
        images_with_captions = []
        
        if not search_results or not search_results.get('metadatas'):
            return images_with_captions
        
        distances = search_results.get('distances', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        
        for i, metadata in enumerate(metadatas[:6]):  # Top 6 matches
            if metadata:
                try:
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity_score = max(0.0, 1 - distance)  # force negative similiarity to zero
                    person_id = metadata.get('person_id', 'Unknown')
                    
                    # Load thumbnail or create placeholder
                    if 'thumbnail' in metadata and metadata['thumbnail']:
                        thumbnail_data = base64.b64decode(metadata['thumbnail'])
                        img = Image.open(io.BytesIO(thumbnail_data))
                    else:
                        img = self.create_placeholder_image(person_id)
                    
                    caption = f"{person_id}\n  {similarity_score:.1%} match"
                    images_with_captions.append((img, caption))
                    
                except Exception as e:
                    print(f"Error loading thumbnail: {e}")
                    placeholder = self.create_placeholder_image("Error")
                    images_with_captions.append((placeholder, "Error loading"))
        
        return images_with_captions
    
    def create_placeholder_image(self, text: str) -> Image.Image:
        """Create a placeholder image with text"""
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), text[:10], fill=(255, 255, 255))
        return img
    
    def get_database_stats(self) -> str:
        """Get current database statistics"""
        try:
            count = self.predictor.db.get_count()
            return f"📊 **Database Statistics**\n\nTotal Faces: {count:,}"
        except Exception as e:
            return f"Error getting stats: {str(e)}"


# health check endpoint, if needed
def health_check():
    """Simple health check function"""
    try:
        return {"status": "healthy", "service": "face-recognition"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Create the Gradio interface
def create_gradio_app():
    """Create and configure the Gradio app"""
    
    # Initialize the app
    app = FaceRecognitionApp()
    
    # Create the interface using Blocks for better layout control
    with gr.Blocks(title="Face Recognition System", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🎭 Face Recognition System
            Upload a face image to check for duplicates in the database.
            
            ## How to use:
            1. Upload a face image using the upload button or drag & drop
            2. Enter the person's name (this will be their ID in the database)
            3. Optionally check "Add to database" to save new faces
            4. Click "Check for Duplicates" to process
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(
                    label="Upload Face Image",
                    type="filepath",
                    height=300
                )
                
                person_name_input = gr.Textbox(
                    label="Person Name (ID)",
                    placeholder="Enter the person's name...",
                    lines=1
                )
                
                add_to_db_checkbox = gr.Checkbox(
                    label="Add to database if not duplicate",
                    value=False
                )
                
                process_btn = gr.Button(
                    "🔍 Check for Duplicates",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Output components
                status_output = gr.Markdown(label="Status")
                
                gr.Markdown("### 🖼️ Top 6 Matches with Similarity Scores")
                gallery_output = gr.Gallery(
                    label="Similar Faces",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height=400,
                    object_fit="contain",
                    show_download_button=True
                )
                
                with gr.Accordion("📋 Detailed Results", open=False):
                    json_output = gr.JSON(label="Match Details")
        
        # Example section
        with gr.Row():
            gr.Examples(
                examples=[
                    ["./src/asset/example1.jpeg", "Ricky Kambuaya", False],
                    ["./src/asset/example2.jpeg", "Marcelino Ferdinan", True],
                ],
                inputs=[image_input, person_name_input, add_to_db_checkbox],
                label="Example Inputs"
            )
        
        # Event handlers
        process_btn.click(
            fn=app.process_uploaded_image,
            inputs=[image_input, person_name_input, add_to_db_checkbox],
            outputs=[
                status_output,
                gallery_output, 
                json_output
            ]
        )
        
        # Add custom CSS for better styling
        demo.css = """
        #gallery {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        """
    
    return demo


# Simple launcher script
if __name__ == "__main__":
    # Create and launch the app
    demo = create_gradio_app()
    
    # Launch with options
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
    )