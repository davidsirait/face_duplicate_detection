"""
Simple entry point for the application
"""
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_gradio_app
from core.config import Config


def main():
    """Main application entry point"""
    print("Starting Face Recognition System...")
    
    # Load configuration
    config = Config("./config.yaml")
    
    # Create Gradio app
    demo = create_gradio_app()
    
    # Launch
    port = 7860
    print(f"Launching on http://localhost:{port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()