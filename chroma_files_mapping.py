import os
import json
from typing import List, Dict

# Path to the compliance documents folder
COMPLIANCE_DOCS_FOLDER = "compliance_documents"
JSON_FILE_PATH = "chroma_files_mapping.json"


def scan_compliance_documents() -> List[str]:
    """
    Scans the compliance_documents folder and returns a list of file paths.
    """
    if not os.path.exists(COMPLIANCE_DOCS_FOLDER):
        os.makedirs(COMPLIANCE_DOCS_FOLDER)
        print(f"Created folder: {COMPLIANCE_DOCS_FOLDER}")
        return []
    
    file_paths = []
    for root, dirs, files in os.walk(COMPLIANCE_DOCS_FOLDER):
        for file in files:
            # Skip hidden files and system files
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths


def get_next_doc_id(existing_mappings: List[Dict]) -> str:
    """
    Generates the next Doc ID based on existing mappings.
    """
    if not existing_mappings:
        return "Doc1"
    
    # Extract all numeric IDs
    doc_numbers = []
    for mapping in existing_mappings:
        doc_id = mapping.get("doc_id", "")
        if doc_id.startswith("Doc"):
            try:
                num = int(doc_id[3:])
                doc_numbers.append(num)
            except ValueError:
                continue
    
    if doc_numbers:
        next_num = max(doc_numbers) + 1
    else:
        next_num = 1
    
    return f"Doc{next_num}"


def get_collection_name_from_path(file_path: str) -> str:
    """
    Generates a Chroma collection name based on the file path or file type.
    You can customize this logic as needed.
    """
    # Simple approach: use a default collection name or derive from folder structure
    # For this example, we'll use "compliance_collection" as default
    # You can modify this to create different collections based on subdirectories, etc.
    return "compliance_collection"


def create_or_update_mapping() -> List[Dict]:
    """
    Creates or updates the chroma_files_mapping.json file.
    Returns the list of file mappings.
    """
    # Case 1: JSON file doesn't exist
    if not os.path.exists(JSON_FILE_PATH):
        print(f"JSON file not found. Creating new mapping file: {JSON_FILE_PATH}")
        file_paths = scan_compliance_documents()
        mappings = []
        
        for idx, file_path in enumerate(file_paths, start=1):
            mapping = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "chroma_collection_name": get_collection_name_from_path(file_path),
                "doc_id": f"Doc{idx}",
                "uploaded": 0
            }
            mappings.append(mapping)
        
        # Save to JSON
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=4, ensure_ascii=False)
        
        print(f"Created mapping file with {len(mappings)} documents.")
        return mappings
    
    # Case 2: JSON file exists - check for new documents
    print(f"Loading existing mapping file: {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        existing_mappings = json.load(f)
    
    # Get current files in the folder
    current_file_paths = scan_compliance_documents()
    
    # Get existing file paths from the mapping
    existing_file_paths = {mapping["file_path"] for mapping in existing_mappings}
    
    # Find new files
    new_files = [fp for fp in current_file_paths if fp not in existing_file_paths]
    
    if new_files:
        print(f"Found {len(new_files)} new document(s). Updating mapping...")
        
        for file_path in new_files:
            new_mapping = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "chroma_collection_name": get_collection_name_from_path(file_path),
                "doc_id": get_next_doc_id(existing_mappings),
                "uploaded": 0
            }
            existing_mappings.append(new_mapping)
            print(f"  Added: {file_path} -> {new_mapping['doc_id']}")
        
        # Save updated mappings
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_mappings, f, indent=4, ensure_ascii=False)
        
        print(f"Updated mapping file with {len(new_files)} new document(s).")
    else:
        print("No new documents found.")
    
    return existing_mappings


def get_chroma_files_mapping() -> List[Dict]:
    """
    Main function to get the chroma files mapping.
    Creates or updates the JSON file and returns the list.
    """
    return create_or_update_mapping()


# For backward compatibility - create a module-level variable
chroma_files_mapping = get_chroma_files_mapping()


if __name__ == "__main__":
    # Test the functionality
    mappings = get_chroma_files_mapping()
    print(f"\nTotal documents in mapping: {len(mappings)}")
    print(f"Documents pending upload: {sum(1 for m in mappings if m['uploaded'] == 0)}")
