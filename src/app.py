import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# تنظیمات ظاهری
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BrainTumorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Brain Tumor Detector AI 🧠")
        self.geometry("650x750")
        self.resizable(False, False)
        
        self.model = None
        self.file_path = None
        
        # --- اصلاح مهم: ترتیب کلاس‌ها بر اساس حروف الفبا ---
        # اگر دیتاسیت شما پوشه‌های 'no' و 'yes' داشته، ترتیب الفبایی میشه:
        # 0: no (سالم)
        # 1: yes (تومور)
        # 2 به بعد: انواع دیگر اگر 4 کلاسه باشه
        self.class_names = ['No Tumor (Healthy)', 'Tumor Detected', 'Glioma', 'Meningioma', 'Pituitary']

        # --- چیدمان صفحه ---
        self.title_label = ctk.CTkLabel(self, text="Brain MRI Classification System", font=("Roboto", 26, "bold"))
        self.title_label.pack(pady=30)

        # قاب نمایش عکس
        self.image_frame = ctk.CTkFrame(self, width=320, height=320, corner_radius=20)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.img_label = ctk.CTkLabel(self.image_frame, text="No Image Selected", font=("Arial", 14))
        self.img_label.pack(expand=True)

        # دکمه آپلود
        self.btn_upload = ctk.CTkButton(self, text="Upload MRI Image 📂", command=self.upload_image, 
                                      width=220, height=45, font=("Arial", 16, "bold"))
        self.btn_upload.pack(pady=20)

        # دکمه تحلیل
        self.btn_analyze = ctk.CTkButton(self, text="Analyze Image 🔍", command=self.analyze_image,
                                       width=220, height=45, font=("Arial", 16, "bold"), 
                                       fg_color="#3498db", state="disabled")
        self.btn_analyze.pack(pady=10)

        # نمایش نتیجه
        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 24, "bold"))
        self.result_label.pack(pady=15)
        
        self.confidence_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.confidence_label.pack(pady=5)

        # لود کردن مدل
        self.load_ai_model()

    def load_ai_model(self):
        try:
            # نام فایل مدل (هر کدام که موجود بود)
                     # پیدا کردن مسیر دقیق پوشه models نسبت به فایل app.py
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir) # یک مرحله عقب‌تر (ریشه پروژه)
            
            model_names = [
                os.path.join(project_root, 'models', 'best_brain_model.h5'),
                os.path.join(project_root, 'models', 'brain_tumor_model.h5')
            ]

            loaded = False
            
            for name in model_names:
                if os.path.exists(name):
                    self.model = load_model(name)
                    print(f"✅ Model loaded: {name}")
                    self.result_label.configure(text="System Ready ✅", text_color="#2ecc71")
                    loaded = True
                    break
            
            if not loaded:
                self.result_label.configure(text="Error: Model Missing!", text_color="#e74c3c")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.result_label.configure(text="Error Loading Model", text_color="#e74c3c")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.file_path = file_path
            
            img = Image.open(file_path)
            img_resized = img.resize((300, 300)) 
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(300, 300))
            
            self.img_label.configure(image=ctk_img, text="")
            self.btn_analyze.configure(state="normal", fg_color="#3498db") # آبی معمولی قبل از کلیک
            self.result_label.configure(text="Ready to Analyze", text_color="white")
            self.confidence_label.configure(text="")

    def analyze_image(self):
        if not self.model or not self.file_path:
            return

        try:
            # 1. آماده‌سازی عکس
            img = Image.open(self.file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # تغییر سایز به 180 (سایز مدل شما)
            target_size = (180, 180) 
            img = img.resize(target_size)
            
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 2. پیش‌بینی
            predictions = self.model.predict(img_array)
            print(f"Raw Predictions: {predictions}") # دیباگ

            # --- بخش منطق اصلاح شده ---
            
            # حالت باینری (2 کلاسه: no, yes)
            if predictions.shape[1] == 2:
                prob_healthy = predictions[0][0] # احتمال کلاس no
                prob_tumor = predictions[0][1]   # احتمال کلاس yes
                
                # اگر احتمال سالم بودن بیشتر بود
                if prob_healthy > prob_tumor:
                    result_text = "No Tumor (Healthy)"
                    confidence_score = prob_healthy * 100
                    is_healthy = True
                else:
                    result_text = "Tumor Detected"
                    confidence_score = prob_tumor * 100
                    is_healthy = False
            
            # حالت 4 کلاسه (اگر مدل 4 کلاسه بود)
            else:
                predicted_index = np.argmax(predictions)
                confidence_score = np.max(predictions) * 100
                
                # فرض بر این است که نام کلاس‌ها را درست چیده باشیم
                # اگر باز هم اشتباه بود، اینجا را باید تغییر داد
                class_names_4 = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] 
                result_text = class_names_4[predicted_index]
                
                if "No Tumor" in result_text:
                    is_healthy = True
                else:
                    is_healthy = False

            # 3. تغییر رنگ و متن بر اساس نتیجه نهایی
            if is_healthy:
                self.result_label.configure(text=f"✅ {result_text}", text_color="#2ecc71") # سبز
                self.btn_analyze.configure(fg_color="#2ecc71")
            else:
                self.result_label.configure(text=f"⚠️ {result_text}", text_color="#e74c3c") # قرمز
                self.btn_analyze.configure(fg_color="#e74c3c")

            self.confidence_label.configure(text=f"Confidence: {confidence_score:.2f}%")

        except Exception as e:
            print(f"❌ Analysis Error: {e}")
            self.result_label.configure(text="Analysis Failed", text_color="orange")

if __name__ == "__main__":
    app = BrainTumorApp()
    app.mainloop()
