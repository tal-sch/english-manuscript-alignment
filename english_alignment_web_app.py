import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. SIAMESE NETWORK ARCHITECTURE
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output

# ==========================================
# 2. SYSTEM SETUP & MODEL LOADING
# ==========================================
# --- UPDATE THESE PATHS IF NEEDED ---
YOLO_WEIGHTS = r"runs\detect\iam_word_slicer\weights\best.pt"
SIAMESE_WEIGHTS = r"siamese_iam_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models to {device}...")

try:
    yolo_model = YOLO(YOLO_WEIGHTS)
    siamese_model = SiameseNetwork().to(device)
    siamese_model.load_state_dict(torch.load(SIAMESE_WEIGHTS, map_location=device))
    siamese_model.eval()
    print("Models successfully loaded for Web App!")
except Exception as e:
    print(f"Error loading models. Check paths. Error: {e}")

siamese_transforms = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================
def extract_and_sort_words(image_path, conf_threshold, imgsz):
    results = yolo_model.predict(source=image_path, conf=conf_threshold, imgsz=int(imgsz), verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() 
    if len(boxes) == 0: return []
    
    # LTR SORTING (English): Sort boxes by the x1 coordinate (left edge) in ASCENDING order
    sorted_indices = np.argsort(boxes[:, 0])
    sorted_boxes = boxes[sorted_indices]
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    crops = []
    for box in sorted_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        crops.append(Image.fromarray(crop).convert("L")) 
    return crops

def compute_similarity_matrix(crops_A, crops_B, threshold, match_score, mismatch_penalty):
    rows, cols = len(crops_A), len(crops_B)
    matrix = np.zeros((rows, cols))
    if rows == 0 or cols == 0: return matrix
        
    tensors_A = torch.stack([siamese_transforms(c) for c in crops_A]).to(device)
    tensors_B = torch.stack([siamese_transforms(c) for c in crops_B]).to(device)
    
    with torch.no_grad():
        vecs_A = siamese_model.forward_once(tensors_A)
        vecs_B = siamese_model.forward_once(tensors_B)
        
        for i in range(rows):
            for j in range(cols):
                dist = F.pairwise_distance(vecs_A[i].unsqueeze(0), vecs_B[j].unsqueeze(0)).item()
                matrix[i][j] = match_score if dist < threshold else mismatch_penalty
    return matrix

def smith_waterman(similarity_matrix, gap_penalty):
    rows, cols = similarity_matrix.shape
    if rows == 0 or cols == 0: return [], [], np.zeros((1,1))
        
    score_matrix = np.zeros((rows + 1, cols + 1))
    max_score = 0
    max_pos = None
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            match_score = score_matrix[i-1][j-1] + similarity_matrix[i-1][j-1]
            delete_score = score_matrix[i-1][j] + gap_penalty
            insert_score = score_matrix[i][j-1] + gap_penalty
            
            score = max(0, match_score, delete_score, insert_score)
            score_matrix[i][j] = score
            if score > max_score:
                max_score = score
                max_pos = (i, j)
                
    align_A, align_B = [], []
    i, j = max_pos if max_pos else (0, 0)
    
    while score_matrix[i][j] != 0:
        current_score = score_matrix[i][j]
        diag = score_matrix[i-1][j-1]
        up = score_matrix[i-1][j]
        
        if current_score == diag + similarity_matrix[i-1][j-1]:
            align_A.append(f"Word {i}")
            align_B.append(f"Word {j}")
            i, j = i-1, j-1
        elif current_score == up + gap_penalty:
            align_A.append(f"Word {i}")
            align_B.append("- (Gap)")
            i -= 1
        else:
            align_A.append("- (Gap)")
            align_B.append(f"Word {j}")
            j -= 1
            
    return align_A[::-1], align_B[::-1]

def generate_plot(crops_A, crops_B, aligned_A, aligned_B):
    n = len(aligned_A)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No Alignment Found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(2, n, figsize=(min(2.5 * n, 20), 4))
    if n == 1: axes = axes.reshape(2, 1)

    for idx in range(n):
        word_A_label, word_B_label = aligned_A[idx], aligned_B[idx]
        
        ax_A = axes[0, idx]
        if "Word" in word_A_label:
            ax_A.imshow(crops_A[int(word_A_label.split()[1]) - 1], cmap='gray')
            ax_A.set_title(f"A: {word_A_label}", color='blue')
        else:
            ax_A.text(0.5, 0.5, "GAP", ha='center', va='center', color='red', weight='bold')
        ax_A.axis('off')
        
        ax_B = axes[1, idx]
        if "Word" in word_B_label:
            ax_B.imshow(crops_B[int(word_B_label.split()[1]) - 1], cmap='gray')
            ax_B.set_title(f"B: {word_B_label}", color='green')
        else:
            ax_B.text(0.5, 0.5, "GAP", ha='center', va='center', color='red', weight='bold')
        ax_B.axis('off')
        
    plt.tight_layout()
    return fig

# ==========================================
# 4. GRADIO INTERFACE LOGIC
# ==========================================
def process_alignment(img_A_path, img_B_path, yolo_conf, yolo_imgsz, threshold, match, mismatch, gap):
    if not img_A_path or not img_B_path:
        return None, "Please upload both images."
        
    crops_A = extract_and_sort_words(img_A_path, yolo_conf, yolo_imgsz)
    crops_B = extract_and_sort_words(img_B_path, yolo_conf, yolo_imgsz)
    
    if not crops_A or not crops_B:
        return None, f"Error: YOLO failed to detect words. Try lowering YOLO Confidence or increasing Image Size. (A: {len(crops_A) if crops_A else 0}, B: {len(crops_B) if crops_B else 0})"
        
    sim_matrix = compute_similarity_matrix(crops_A, crops_B, threshold, match, mismatch)
    aligned_A, aligned_B = smith_waterman(sim_matrix, gap)
    
    fig = generate_plot(crops_A, crops_B, aligned_A, aligned_B)
    
    text_out = f"Found {len(crops_A)} words in Line A, and {len(crops_B)} words in Line B.\n\n"
    for a, b in zip(aligned_A, aligned_B):
        text_out += f"Line A: {a:<12} | Line B: {b}\n"
        
    return fig, text_out

# ==========================================
# 5. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 📜 English Handwriting Aligner
        Upload two image snippets of English handwriting. The AI will use YOLO to slice the words, a Siamese Network to compare them, and the Smith-Waterman algorithm to find the best local alignment.
        """
    )
    
    with gr.Row():
        with gr.Column():
            img_a = gr.Image(type="filepath", label="Line A (Copy 1)")
            img_b = gr.Image(type="filepath", label="Line B (Copy 2)")
            align_btn = gr.Button("🚀 Run Alignment Pipeline", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Algorithm Hyperparameters")
            yolo_conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="YOLO Confidence Threshold (Lower if words aren't detected)")
            yolo_imgsz = gr.Slider(320, 2048, value=640, step=32, label="YOLO Image Size (Increase for very wide images)")
            threshold = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Siamese Match Threshold (Lower = stricter)")
            match_score = gr.Slider(1, 5, value=2, step=1, label="Match Score Reward")
            mismatch_pen = gr.Slider(-5, 0, value=-1, step=1, label="Mismatch Penalty")
            gap_pen = gr.Slider(-5, 0, value=-2, step=1, label="Gap Penalty")
            
            raw_text = gr.Textbox(label="Raw Alignment Mapping", lines=8)
            
    with gr.Row():
        output_plot = gr.Plot(label="Visual Alignment Output")

    align_btn.click(
        fn=process_alignment,
        inputs=[img_a, img_b, yolo_conf, yolo_imgsz, threshold, match_score, mismatch_pen, gap_pen],
        outputs=[output_plot, raw_text]
    )

if __name__ == "__main__":
    app.launch()
