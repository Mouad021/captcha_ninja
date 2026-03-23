# -*- coding: utf-8 -*-
"""
====================================================================================================
    PROJECT NINJA CAPTCHA AI - MAX SPEED & CNN EDITION (PyTorch)
====================================================================================================
"""

import os
import cv2
import base64
import orjson
import socket
import time
import random
import threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# --- استيراد PyTorch للذكاء الاصطناعي الحقيقي ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# إعداد المعالج (استخدام CPU كافٍ جداً ومناسب لـ Railway)
device = torch.device("cpu")

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ==========================================================
# 👁️ دالة الرؤية المحسنة (Computer Vision)
# ==========================================================
def process_full_image(b64_string):
    """
    تقوم بتنظيف الصورة، إزالة خطوط الشطب (Strikethrough)، والتحجيم القياسي 64x32
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        raw = base64.b64decode(b64_string)
        np_arr = np.frombuffer(raw, np.uint8)
        
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 1. تحويل لرمادي وتطبيق عتبة تكيفية (لفصل النص عن الخلفية الملونة)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        # 2. إزالة خطوط الشطب الأفقية (Underlines & Strikethroughs) بذكاء
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, 0, 3) # مسح الخط باللون الأسود
            
        # 3. تنظيف النقاط المزعجة
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 4. القص والتحجيم (Cropping & Resizing)
        coords = cv2.findNonZero(cleaned)
        TARGET_W, TARGET_H = 64, 32  # أبعاد قياسية للـ CNN
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = cleaned[max(0, y-2):min(cleaned.shape[0], y+h+2), 
                              max(0, x-2):min(cleaned.shape[1], x+w+2)]
        else:
            cropped = cleaned
            
        # وضع الصورة المقصوصة في وسط قماش أسود
        scale = min(TARGET_W / max(1, cropped.shape[1]), TARGET_H / max(1, cropped.shape[0]))
        new_w, new_h = max(1, int(cropped.shape[1] * scale)), max(1, int(cropped.shape[0] * scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        x_offset = (TARGET_W - new_w) // 2
        y_offset = (TARGET_H - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # التطبيع (Normalization)
        tensor_img = canvas.astype(np.float32) / 255.0
        return tensor_img # Shape: (32, 64)
    except Exception as e:
        print(f"\033[91m [!] VISION ERROR: {e} \033[0m")
        return None

# ==========================================================
# 🧠 بنية الشبكة التلافيفية (CNN Architecture)
# ==========================================================
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        # استخراج الميزات (Features Extraction)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # الشبكة العصبية العميقة (Dense Layers)
        self.fc1 = nn.Linear(64 * 8 * 16, 256)
        self.dropout = nn.Dropout(0.2)
        
        # 3 مخارج لكل رقم (من 0 إلى 9)
        self.out1 = nn.Linear(256, 10)
        self.out2 = nn.Linear(256, 10)
        self.out3 = nn.Linear(256, 10)

    def forward(self, x):
        # x shape: (Batch, 1, 32, 64)
        x = self.pool(F.relu(self.conv1(x))) # -> (Batch, 32, 16, 32)
        x = self.pool(F.relu(self.conv2(x))) # -> (Batch, 64, 8, 16)
        x = x.view(x.size(0), -1)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out1(x), self.out2(x), self.out3(x)

# ==========================================================
# 🤖 إدارة العقل والذاكرة (Brain Manager)
# ==========================================================
class NinjaShapeBrain:
    def __init__(self):
        self.model = CaptchaCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.memory_file = os.path.join(current_dir, "ninja_cnn_memory.pth")
        
        self.memory = []
        self.max_memory = 20000 
        self.lock = threading.Lock()
        self.learning_steps = 0
        
        self.load_brain()

    def remember(self, shape_pixels, target_number_str):
        if len(self.memory) > self.max_memory: self.memory.pop(0)
        self.memory.append((shape_pixels, target_number_str))

    def predict_batch(self, X_batch_np):
        self.model.eval() # وضع التوقع
        with torch.no_grad():
            # تحويل Numpy إلى PyTorch Tensor
            x_tensor = torch.tensor(X_batch_np, dtype=torch.float32).unsqueeze(1).to(device)
            out1, out2, out3 = self.model(x_tensor)
            
            p1 = torch.argmax(out1, dim=1).cpu().numpy()
            p2 = torch.argmax(out2, dim=1).cpu().numpy()
            p3 = torch.argmax(out3, dim=1).cpu().numpy()
            
            return [f"{a}{b}{c}" for a, b, c in zip(p1, p2, p3)]

    def train_step(self, mini_batch):
        self.model.train() # وضع التدريب
        X_list, Y1_list, Y2_list, Y3_list = [], [], [], []
        
        for img, label in mini_batch:
            # إضافة اهتزاز بسيط (Data Augmentation) لتسريع التعلم
            shift_x, shift_y = random.randint(-1, 1), random.randint(-1, 1)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            aug_img = cv2.warpAffine(img, M, (64, 32))
            
            X_list.append(aug_img)
            Y1_list.append(int(label[0]))
            Y2_list.append(int(label[1]))
            Y3_list.append(int(label[2]))
            
        x_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(1).to(device)
        y1_tensor = torch.tensor(Y1_list, dtype=torch.long).to(device)
        y2_tensor = torch.tensor(Y2_list, dtype=torch.long).to(device)
        y3_tensor = torch.tensor(Y3_list, dtype=torch.long).to(device)
        
        self.optimizer.zero_grad()
        out1, out2, out3 = self.model(x_tensor)
        
        loss = self.criterion(out1, y1_tensor) + self.criterion(out2, y2_tensor) + self.criterion(out3, y3_tensor)
        loss.backward()
        self.optimizer.step()
        
        self.learning_steps += 1
        
        # حساب الدقة
        p1, p2, p3 = torch.argmax(out1, dim=1), torch.argmax(out2, dim=1), torch.argmax(out3, dim=1)
        correct = ((p1 == y1_tensor) & (p2 == y2_tensor) & (p3 == y3_tensor)).sum().item()
        
        return loss.item(), correct

    def save_brain(self):
        with self.lock:
            torch.save({
                'steps': self.learning_steps,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.memory_file)

    def load_brain(self):
        if os.path.exists(self.memory_file):
            checkpoint = torch.load(self.memory_file, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_steps = checkpoint.get('steps', 0)
            print("\033[92m [🧠] PyTorch CNN BRAIN LOADED SUCCESSFULLY! \033[0m")
        else:
            print("\033[93m [👶] EMPTY BRAIN: Ready to learn dynamic shapes! \033[0m")

BRAIN = NinjaShapeBrain()

# ==========================================================
# 🌐 إعدادات الخادم والـ API
# ==========================================================
app = FastAPI(title="NINJA CAPTCHA AI - PRO")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/")
async def lawissi_endpoint(req: Request):
    """
    نقطة النهاية (Endpoint) التي يتصل بها السكريبت الخاص بك لحل الكابتشا
    """
    t0 = time.time()
    raw_body = await req.body()
    try: payload = orjson.loads(raw_body)
    except: return Response(content=b'{"status":"error"}', media_type="application/json")

    target_number = str(payload.get("n", "")).strip()
    images_data = payload.get("images", [])
    
    if not target_number or not images_data:
        return Response(content=b'{"status":"error"}', media_type="application/json")

    valid_pixels = []
    valid_indices = []
    
    for idx_str, b64_img in enumerate(images_data):
        shape_pixels = process_full_image(b64_img)
        if shape_pixels is not None:
            valid_pixels.append(shape_pixels)
            valid_indices.append(idx_str)

    results = []
    if valid_pixels:
        X_batch = np.array(valid_pixels)
        batch_predictions = BRAIN.predict_batch(X_batch)
        
        for idx, pred in zip(valid_indices, batch_predictions):
            # نرسل النتيجة كـ ok فقط إذا طابق الهدف (لتقليل الضغط على السكريبت)
            if pred == target_number:
                results.append({
                    "status": "ok",
                    "index": int(idx),
                    "recognized_text": pred
                })

    response_data = {
        "status": "success",
        "results": results
    }

    dur = (time.time() - t0) * 1000
    print(f"\033[92m [⚡] Target: {target_number} | Found: {len(results)} | Latency: {dur:.1f}ms \033[0m")
    return Response(content=orjson.dumps(response_data), media_type="application/json")


@app.post("/ninja_learn")
async def ninja_learn(req: Request):
    """
    نقطة النهاية لتلقيم الذكاء الاصطناعي بالصور الصحيحة ليتعلم منها
    """
    raw_body = await req.body()
    try: payload = orjson.loads(raw_body)
    except: return Response(content=b"Failed", status_code=400)

    target_number = str(payload.get("target_number")).strip()
    images_b64 = payload.get("images", [])

    if not target_number or len(target_number) != 3 or not images_b64:
        return Response(content=b"Missing Data", status_code=400)

    saved_count = 0
    for img_str in images_b64:
        shape_pixels = process_full_image(img_str)
        if shape_pixels is not None:
            with BRAIN.lock:
                BRAIN.remember(shape_pixels, target_number)
            saved_count += 1

    print(f"\033[94m [📥] INJECTED {saved_count} SAMPLES TO MEMORY (Total: {len(BRAIN.memory)}) \033[0m")
    return Response(content=b"Saved Successfully")


def background_training_loop():
    """
    حلقة تدريب خلفية تتعلم باستمرار من الذاكرة لتتكيف مع الأشكال الجديدة
    """
    batch_size = 64 
    last_save_step = int(BRAIN.learning_steps)
    
    while True:
        with BRAIN.lock:
            mem_size = len(BRAIN.memory)
            
        if mem_size >= batch_size:
            with BRAIN.lock:
                mini_batch = random.sample(BRAIN.memory, batch_size)
            
            loss, correct_guesses = BRAIN.train_step(mini_batch)
                
            if BRAIN.learning_steps - last_save_step >= 200: 
                BRAIN.save_brain()
                last_save_step = int(BRAIN.learning_steps)
                print("\033[92m [💾] CNN BRAIN AUTO-SAVED! \033[0m")

            acc = (correct_guesses / batch_size) * 100
            print(f"\033[96m [🏋️] TRAINING | Loss: {loss:.4f} | Acc: {acc:.0f}% | Steps: {BRAIN.learning_steps} \033[0m")
            time.sleep(0.05) 
        else:
            time.sleep(2)

if __name__ == '__main__':
    os.system('color') 
    threading.Thread(target=background_training_loop, daemon=True).start()

    server_ip = get_lan_ip()
    print("===========================================================")
    print(" NINJA CAPTCHA AI | PyTorch CNN MAX SPEED EDITION")
    print("===========================================================")
    print(f" [\033[92m📡\033[0m] SERVER IP   : {server_ip}")
    print(f" [\033[92m🔌\033[0m] SERVER PORT : 5000")
    print("===========================================================")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1, log_level="critical", access_log=False)
