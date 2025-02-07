import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import pytesseract
import pdf2image
import fitz
import cv2
import numpy as np
import os

class OCRViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Document Viewer")
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.select_button = ttk.Button(
            self.main_frame,
            text="Select PDF/Image",
            command=self.select_file
        )
        self.select_button.pack(pady=10)
        
        self.create_display_windows()
        self.text_boxes = []  # Store references to text boxes
        
    def create_display_windows(self):
        # Create windows
        self.windows = {
            'original': self.create_window("Original Document"),
            'overlay': self.create_window("Interactive Text Overlay"),
            'corrected': self.create_window("Corrected Text Overlay")
        }
        
        # Create canvases
        for name, window in self.windows.items():
            # Create a frame with scrollbars
            frame = ttk.Frame(window)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Create canvas with scrollbars
            canvas = tk.Canvas(frame, bg='white')
            v_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            
            # Configure canvas scrolling
            canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
            
            # Grid layout for canvas and scrollbars
            canvas.grid(row=0, column=0, sticky="nsew")
            v_scroll.grid(row=0, column=1, sticky="ns")
            h_scroll.grid(row=1, column=0, sticky="ew")
            
            # Configure grid weights
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            
            setattr(self, f"{name}_canvas", canvas)
            
            # Bind mouse wheel events
            canvas.bind('<Configure>', lambda e, c=canvas: self.on_canvas_configure(c))
            canvas.bind_all("<MouseWheel>", lambda e, c=canvas: self.on_mousewheel(e, c))
    
    def create_window(self, title):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("800x600")
        return window
    
    def on_canvas_configure(self, canvas):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    def on_mousewheel(self, event, canvas):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF and Image files", "*.pdf *.png *.jpg *.jpeg *.tiff *.bmp")]
        )
        if file_path:
            self.process_file(file_path)
    
    def process_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Clear existing text boxes
        self.clear_text_boxes()
        
        if file_extension == '.pdf':
            self.process_pdf(file_path)
        else:
            self.process_image(file_path)
    
    def clear_text_boxes(self):
        for box in self.text_boxes:
            box.destroy()
        self.text_boxes = []
    
    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        page = doc[0]  # Process first page for demonstration
        
        # Get page as image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Display original image
        self.display_image(img)
        
        # Get words with their positions
        words = page.get_text("words")
        self.create_text_overlays(words)
    
    def process_image(self, image_path):
        img = Image.open(image_path)
        
        # Display original image
        self.display_image(img)
        
        # Perform OCR with position information
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        self.create_text_overlays_from_ocr(data)
    
    def display_image(self, img):
        # Resize image while maintaining aspect ratio
        display_width = 780
        ratio = display_width / img.width
        display_height = int(img.height * ratio)
        self.scale_factor = ratio
        
        img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Display in all windows
        for name in ['original', 'overlay', 'corrected']:
            canvas = getattr(self, f"{name}_canvas")
            photo = ImageTk.PhotoImage(img_resized)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            setattr(self, f"{name}_photo", photo)
            
            # Update scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
    
    def create_text_overlays(self, words):
        for word in words:
            x0, y0, x1, y1, text = word[:5]
            
            # Scale coordinates according to display size
            x0 *= self.scale_factor
            y0 *= self.scale_factor
            x1 *= self.scale_factor
            y1 *= self.scale_factor
            
            # Create text entry for overlay window
            text_box = tk.Entry(self.overlay_canvas, bd=0, highlightthickness=0)
            text_box.insert(0, text)
            text_box.configure(width=len(text))
            
            # Position text box on canvas
            self.overlay_canvas.create_window(x0, y0, window=text_box, anchor="nw")
            self.text_boxes.append(text_box)
            
            # Create similar text box for corrected window
            corrected_box = tk.Entry(self.corrected_canvas, bd=0, highlightthickness=0)
            corrected_box.insert(0, text)
            corrected_box.configure(width=len(text))
            
            # Position text box on canvas
            self.corrected_canvas.create_window(x0, y0, window=corrected_box, anchor="nw")
            self.text_boxes.append(corrected_box)
    
    def create_text_overlays_from_ocr(self, data):
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Filter low-confidence results
                x = data['left'][i] * self.scale_factor
                y = data['top'][i] * self.scale_factor
                text = data['text'][i]
                
                if text.strip():  # Only process non-empty text
                    # Create text entry for overlay window
                    text_box = tk.Entry(self.overlay_canvas, bd=0, highlightthickness=0)
                    text_box.insert(0, text)
                    text_box.configure(width=len(text))
                    
                    # Position text box on canvas
                    self.overlay_canvas.create_window(x, y, window=text_box, anchor="nw")
                    self.text_boxes.append(text_box)
                    
                    # Create similar text box for corrected window
                    corrected_box = tk.Entry(self.corrected_canvas, bd=0, highlightthickness=0)
                    corrected_box.insert(0, text)
                    corrected_box.configure(width=len(text))
                    
                    # Position text box on canvas
                    self.corrected_canvas.create_window(x, y, window=corrected_box, anchor="nw")
                    self.text_boxes.append(corrected_box)

def main():
    root = tk.Tk()
    app = OCRViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
