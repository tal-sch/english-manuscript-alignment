import os
import cv2
import glob
import random
import yaml
from lxml import etree
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data"
OUTPUT_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\YOLO_Dataset"
PATCH_SIZE = 320
STRIDE = 250
TRAIN_SPLIT = 0.8

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def parse_xml_for_words(xml_path):
    """Parses an IAM XML file and returns a list of word bounding boxes [x1, y1, x2, y2]."""
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    words_bboxes = []
    
    # Iterate over all word elements
    for word in root.xpath('.//word'):
        # A word might consist of multiple connected components (cmp)
        cmps = word.findall('cmp')
        if not cmps:
            continue
            
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for cmp in cmps:
            x = int(cmp.get('x'))
            y = int(cmp.get('y'))
            w = int(cmp.get('width'))
            h = int(cmp.get('height'))
            x_mins.append(x)
            y_mins.append(y)
            x_maxs.append(x + w)
            y_maxs.append(y + h)
            
        if x_mins:
            word_bbox = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
            words_bboxes.append(word_bbox)
            
    return words_bboxes

def get_boxes_in_patch(words_bboxes, px, py, patch_size):
    """Finds words that fall inside the current patch and adjusts their coordinates."""
    patch_boxes = []
    
    for box in words_bboxes:
        x1, y1, x2, y2 = box
        
        # Check if the center of the word is inside the patch
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if px <= cx < px + patch_size and py <= cy < py + patch_size:
            # Clip bounding box to patch boundaries
            nx1 = max(x1, px) - px
            ny1 = max(y1, py) - py
            nx2 = min(x2, px + patch_size) - px
            ny2 = min(y2, py + patch_size) - py
            
            # Avoid empty/invalid boxes after clipping
            if nx2 > nx1 and ny2 > ny1:
                patch_boxes.append([nx1, ny1, nx2, ny2])
                
    return patch_boxes

def process_form(xml_path, split):
    """Processes a single form: creates patches and saves YOLO labels."""
    form_id = os.path.splitext(os.path.basename(xml_path))[0]
    
    # Try finding the corresponding image
    img_path = os.path.join(DATA_DIR, 'forms', f"{form_id}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(DATA_DIR, 'forms', f"{form_id}.jpg")
    
    if not os.path.exists(img_path):
        # Could be inside subfolders if extracted directly, but we extracted flat
        # Try to search recursively if not found
        found = list(Path(DATA_DIR).rglob(f"{form_id}.*"))
        if found:
            img_path = str(found[0])
        else:
            print(f"Warning: Image for {form_id} not found.")
            return
            
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return
        
    h_img, w_img = img.shape[:2]
    words_bboxes = parse_xml_for_words(xml_path)
    
    patch_idx = 0
    # Sliding window
    for py in range(0, h_img - PATCH_SIZE + STRIDE, STRIDE):
        for px in range(0, w_img - PATCH_SIZE + STRIDE, STRIDE):
            # Ensure we don't go out of bounds
            py_actual = min(py, h_img - PATCH_SIZE)
            px_actual = min(px, w_img - PATCH_SIZE)
            
            if py_actual < 0 or px_actual < 0:
                continue
                
            patch_boxes = get_boxes_in_patch(words_bboxes, px_actual, py_actual, PATCH_SIZE)
            
            # Only save patch if it contains at least one word
            if len(patch_boxes) > 0:
                patch = img[py_actual:py_actual+PATCH_SIZE, px_actual:px_actual+PATCH_SIZE]
                
                # Create label content
                label_lines = []
                for box in patch_boxes:
                    nx1, ny1, nx2, ny2 = box
                    
                    # YOLO normalized format
                    cx = ((nx1 + nx2) / 2) / PATCH_SIZE
                    cy = ((ny1 + ny2) / 2) / PATCH_SIZE
                    w = (nx2 - nx1) / PATCH_SIZE
                    h = (ny2 - ny1) / PATCH_SIZE
                    
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                # Save patch and label
                patch_name = f"{form_id}_{patch_idx}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', split, f"{patch_name}.jpg"), patch)
                
                with open(os.path.join(OUTPUT_DIR, 'labels', split, f"{patch_name}.txt"), 'w') as f:
                    f.write('\n'.join(label_lines))
                    
                patch_idx += 1

def generate_yaml():
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'word'
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def main():
    setup_directories()
    
    xml_files = glob.glob(os.path.join(DATA_DIR, 'xml', '*.xml'))
    if not xml_files:
        print(f"No XML files found in {os.path.join(DATA_DIR, 'xml')}")
        return
        
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * TRAIN_SPLIT)
    train_xmls = xml_files[:split_idx]
    val_xmls = xml_files[split_idx:]
    
    print(f"Found {len(xml_files)} forms. Processing {len(train_xmls)} for train, {len(val_xmls)} for val...")
    
    for xml_path in tqdm(train_xmls, desc="Processing Train Forms"):
        process_form(xml_path, 'train')
        
    for xml_path in tqdm(val_xmls, desc="Processing Val Forms"):
        process_form(xml_path, 'val')
        
    generate_yaml()
    print("\nDataset preparation complete! You can now run the YOLO training script.")

if __name__ == "__main__":
    main()
