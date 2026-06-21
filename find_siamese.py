import json

def main():
    notebook_path = r"c:\Users\Tal Sch\Desktop\manuscript_alignment\Notebook.ipynb"
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if "nn.Module" in source or "Siamese" in source:
                print(f"--- Cell {i} ---")
                print(source)
                print("----------------\n")

if __name__ == "__main__":
    main()
