import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
from language_tool_python import LanguageTool

class OCRWorker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, image):
        super().__init__()
        self.image = image
        
    def run(self):
        try:
            # Ensure image is in the correct format for Tesseract
            if isinstance(self.image, np.ndarray):
                # Convert to grayscale if it's not already
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image
                # Improve image quality for OCR
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(gray)
            else:
                raise ValueError(f"Unsupported image type: {type(self.image)}")
            self.finished.emit(text)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document OCR Viewer")
        self.setMinimumSize(1200, 800)
        
        # Initialize language tool for grammar correction
        self.language_tool = LanguageTool('en-US')
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add upload button
        upload_btn = QPushButton("Upload Document")
        upload_btn.clicked.connect(self.upload_document)
        layout.addWidget(upload_btn)
        
        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        # Create viewers container
        viewers_container = QHBoxLayout()
        
        # Create and set up viewers similar to before...
        self.setup_viewers(viewers_container)
        
        layout.addLayout(viewers_container)

    def setup_viewers(self, container):
        # Original document viewer
        self.original_viewer = QScrollArea()
        self.original_viewer.setWidgetResizable(True)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_viewer.setWidget(self.original_label)
        
        # OCR text viewer
        self.ocr_viewer = QScrollArea()
        self.ocr_viewer.setWidgetResizable(True)
        self.ocr_text = QLabel()
        self.ocr_text.setWordWrap(True)
        self.ocr_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ocr_viewer.setWidget(self.ocr_text)
        
        # Corrected text viewer
        self.corrected_viewer = QScrollArea()
        self.corrected_viewer.setWidgetResizable(True)
        self.corrected_text = QLabel()
        self.corrected_text.setWordWrap(True)
        self.corrected_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.corrected_viewer.setWidget(self.corrected_text)
        
        # Add viewers with labels
        for viewer, title in [
            (self.original_viewer, "Original Document"),
            (self.ocr_viewer, "OCR Text"),
            (self.corrected_viewer, "Corrected Text")
        ]:
            layout = QVBoxLayout()
            layout.addWidget(QLabel(title))
            layout.addWidget(viewer)
            container.addLayout(layout)
    
    def process_pdf(self, file_path):
        try:
            self.status_label.setText("Processing PDF...")
            
            # Open PDF
            doc = fitz.open(file_path)
            page = doc[0]
            
            # Convert PDF page to image with higher resolution
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to numpy array
            # Get image data as bytes
            img_data = bytes(pix.samples)
            
            # Create numpy array from bytes
            np_arr = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            
            # Display original
            height, width, channel = np_arr.shape
            bytes_per_line = 3 * width
            q_img = QImage(np_arr.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            scaled_pixmap = pixmap.scaled(
                self.original_viewer.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.original_label.setPixmap(scaled_pixmap)
            
            # Start OCR in separate thread
            self.ocr_worker = OCRWorker(np_arr)
            self.ocr_worker.finished.connect(self.handle_ocr_result)
            self.ocr_worker.start()
            
            doc.close()
            self.status_label.setText("PDF processing initiated...")
            
        except Exception as e:
            self.status_label.setText(f"Error processing PDF: {str(e)}")
            print(f"Detailed error: {str(e)}")
    
    def process_image(self, file_path):
        try:
            self.status_label.setText("Processing image...")
            # Read image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Convert to RGB for display
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Display image
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(
                self.original_viewer.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.original_label.setPixmap(scaled_pixmap)
            
            # Process OCR
            self.ocr_worker = OCRWorker(img)
            self.ocr_worker.finished.connect(self.handle_ocr_result)
            self.ocr_worker.start()
            
            self.status_label.setText("Image processing initiated...")
            
        except Exception as e:
            self.status_label.setText(f"Error processing image: {str(e)}")
            print(f"Detailed error: {str(e)}")
    
    def handle_ocr_result(self, text):
        try:
            if text.startswith("Error:"):
                self.status_label.setText(text)
                return
                
            # Display OCR text
            self.ocr_text.setText(text)
            
            # Perform grammar correction
            corrected = self.language_tool.correct(text)
            self.corrected_text.setText(corrected)
            self.status_label.setText("Processing completed successfully!")
        except Exception as e:
            self.status_label.setText(f"Error in post-processing: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Documents (*.pdf *.png *.jpg *.jpeg *.tiff)"
        )
        
        if file_path:
            if file_path.lower().endswith('.pdf'):
                self.process_pdf(file_path)
            else:
                self.process_image(file_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
