import sys
from db import FaceVectorDB

def main():
    # Initialize the database
    db = FaceVectorDB(persist_directory="./face_db", collection_name="face_embeddings")
    
    # Get count before clearing
    count = db.get_count()
    print(f"Found {count} embeddings in database")
    
    # Clear the database
    print("Clearing database...")
    db.clear()
    
    # Verify it's empty
    new_count = db.get_count()
    print(f"✓ Database cleared. Current count: {new_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
