import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Tuple, List, Optional, Dict
import json
import os
import shutil
import sys
sys.path.append("./src")

# Import your existing modules
from inference import FaceDuplicateDetector
from db import FaceVectorDB
from detector import FaceDetector  # Assuming you have face processing logic

class FaceRecognitionApp:
    def __init__(self, config_path = "./config.yaml"):
        """Initialize the Face Recognition App"""
        self.predictor = FaceDuplicateDetector(config_path=config_path)  # Your face processing class
        
    def process_uploaded_image(
        self, 
        image: str,  
        person_name: str, 
        add_to_db: bool
    ) -> Tuple[str, List[Tuple[Image.Image, str]], str]:
        """
        Process uploaded image and check for duplicates
        
        Args:
            image: Uploaded image as numpy array
            person_name: Name of the person (becomes person_id)
            add_to_db: Whether to add the face to database
            
        Returns:
            Tuple of (status_message, list_of_(image,caption)_tuples, json_details)
        """
        try:
            # Validate inputs
            if image is None:
                return "No image uploaded", [], "{}"
            
            if not person_name or person_name.strip() == "":
                # return "Please provide the person name", [], "{}"
                return "Please provide the person name"
            
            # check for duplicates
            is_duplicate, match_info, top_matches = self.predictor.check_duplicate(image)

            # process top_matches results
            gallery_images_with_captions = self.process_search_results_with_scores(top_matches)
            
            # Prepare status message
            if is_duplicate:
                similarity_score = 1 - match_info['distance']  # Convert distance to similarity
                status_msg = f"**Duplicate Found!**\n"
                status_msg += f"Similarity Score: {similarity_score:.2%}\n"
                status_msg += f"Matched Person: {match_info.get('metadata', {}).get('person_id', 'Unknown')}"
            else:
                status_msg = "**No duplicate found**"
            
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
            return f" Error: {str(e)}", [], "{}"
    
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
                    # Calculate similarity score (1 - distance)
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity_score = 1 - distance
                    person_id = metadata.get('person_id', 'Unknown')
                    
                    # Load thumbnail or create placeholder
                    if 'thumbnail' in metadata and metadata['thumbnail']:
                        thumbnail_data = base64.b64decode(metadata['thumbnail'])
                        img = Image.open(io.BytesIO(thumbnail_data))
                    else:
                        # Create placeholder if no thumbnail
                        img = self.create_placeholder_image(person_id)
                    
                    # Add score overlay to image (optional - you can remove this if you prefer clean images)
                    img = self.add_score_overlay(img, similarity_score)
                    
                    # Create caption with similarity score
                    caption = f"{person_id}\n  {similarity_score:.1%} match"
                    
                    images_with_captions.append((img, caption))
                    
                except Exception as e:
                    print(f"Error loading thumbnail: {e}")
                    placeholder = self.create_placeholder_image("Error")
                    images_with_captions.append((placeholder, "Error loading"))
        
        return images_with_captions
    
    def add_score_overlay(self, img: Image.Image, score: float) -> Image.Image:
        """Add similarity score overlay to image"""
        # Create a copy to avoid modifying original
        img = img.copy()
        draw = ImageDraw.Draw(img)
        
        # Add semi-transparent background for score
        width, height = img.size
        rect_height = 20
        
        # Create overlay rectangle at bottom
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [(0, height - rect_height), (width, height)],
            fill=(0, 0, 0, 180)  # Semi-transparent black
        )
        
        # Composite the overlay
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        # Add text
        draw = ImageDraw.Draw(img)
        text = f"{score:.1%}"
        
        # Try to use a font, fall back to default if not available
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = None
        
        draw.text(
            (5, height - rect_height + 2),
            text,
            fill=(255, 255, 255),
            font=font
        )
        
        return img
    
    def create_placeholder_image(self, text: str) -> Image.Image:
        """Create a placeholder image with text"""
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), text[:10], fill=(255, 255, 255))
        return img
    
    def get_database_stats(self) -> str:
        """Get current database statistics"""
        try:
            count = self.db.get_count()
            return f"📊 **Database Statistics**\n\nTotal Faces: {count:,}"
        except Exception as e:
            return f"Error getting stats: {str(e)}"


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
                
                # Database stats
                with gr.Accordion("Database Info", open=False):
                    stats_text = gr.Markdown(app.get_database_stats())
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
            
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
                    ["example1.jpg", "John Doe", False],
                    ["example2.jpg", "Jane Smith", True],
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
        
        refresh_stats_btn.click(
            fn=app.get_database_stats,
            inputs=[],
            outputs=[stats_text]
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
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set True to create public link
        debug=True,             # Show debug info
        show_error=True,        # Show errors in UI
        # auth=("admin", "password")  # Optional authentication
    )
    
    # For production deployment to Hugging Face Spaces:
    # demo.launch()