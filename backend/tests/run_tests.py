import os
import sys
import json

# Ensure we can import from app
sys.path.append(os.path.abspath(os.curdir))

from app.core.preprocess_image import preprocess_image
from app.core.preprocess_document import preprocess_document

def test_image():
    print("Testing Image Preprocessing...")
    path = "tests/test_image.png"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    
    with open(path, "rb") as f:
        data = f.read()
    
    result = preprocess_image(data)
    print("Image Preprocessing Result (Summary):")
    print(f"- Resolution: {result['width']}x{result['height']}")
    print(f"- Format: {result['format']}")
    print(f"- Perceptual Hash (PHash): {result['context_features']['perceptual_hash_p']}")
    print(f"- ELA Mean: {result['authenticity_features']['ela_mean']}")
    print("Success!\n")

def test_document():
    print("Testing Document Preprocessing...")
    path = "tests/test_doc.txt"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    
    with open(path, "rb") as f:
        data = f.read()
    
    result = preprocess_document(data, "test_doc.txt")
    print("Document Preprocessing Result (Summary):")
    print(f"- Language: {result['context_features']['language']}")
    print(f"- Word Count: {result['source_features']['word_count']}")
    print(f"- Entities found: {len(result['context_features']['entities'])}")
    print(f"- Keywords: {result['context_features']['keywords'][:5]}")
    print("Success!\n")

if __name__ == "__main__":
    test_image()
    test_document()
    print("All tests completed.")
