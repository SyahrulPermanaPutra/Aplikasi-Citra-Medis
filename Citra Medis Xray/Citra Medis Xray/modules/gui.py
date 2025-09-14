import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from pathlib import Path
import cv2  # ✅ TAMBAHKAN IMPORT INI
import numpy as np  # ✅ TAMBAHKAN IMPORT INI

class MedicalImageGUI:
    def __init__(self, root, image_processor):
        self.root = root
        self.processor = image_processor
        self.image_paths = []
        self.current_results = {}
        self.current_image_index = 0
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        self.root.title("Medical Image Enhancement Processor")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="Load Images", command=self.load_images).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Process All", command=self.process_images).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Save Results", command=self.save_results).grid(row=0, column=2, padx=5)
        
        # Image selection
        ttk.Label(control_frame, text="Select Image:").grid(row=0, column=3, padx=(20, 5))
        self.image_var = tk.StringVar()
        self.image_combo = ttk.Combobox(control_frame, textvariable=self.image_var, state='readonly')
        self.image_combo.grid(row=0, column=4, padx=5)
        self.image_combo.bind('<<ComboboxSelected>>', self.on_image_select)
        
        # Technique selection
        ttk.Label(control_frame, text="Technique:").grid(row=0, column=5, padx=(20, 5))
        self.tech_var = tk.StringVar()
        self.tech_combo = ttk.Combobox(control_frame, textvariable=self.tech_var, 
                                      values=list(self.processor.techniques.keys()),
                                      state='readonly')
        self.tech_combo.grid(row=0, column=6, padx=5)
        self.tech_combo.bind('<<ComboboxSelected>>', self.on_tech_select)
        self.tech_combo.current(0)
        
        # Display area
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="5")
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(main_frame, text="Metrics", padding="5")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=100)
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load images")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_images(self):
        """Load multiple images"""
        file_paths = filedialog.askopenfilenames(
            title="Select X-Ray Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.image_paths = list(file_paths)
            self.image_combo['values'] = [Path(path).name for path in self.image_paths]
            self.status_var.set(f"Loaded {len(self.image_paths)} images")
            
            if self.image_paths:
                self.image_combo.current(0)
                self.display_image(self.image_paths[0])
    
    def process_images(self):
        """Process all images in a separate thread"""
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please load images first!")
            return
        
        self.status_var.set("Processing images...")
        
        # Run in separate thread to avoid GUI freezing
        def process_thread():
            self.current_results = self.processor.process_batch(self.image_paths)
            self.root.after(0, self.on_processing_complete)
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def on_processing_complete(self):
        """Called when processing is complete"""
        self.status_var.set(f"Processing complete! Processed {len(self.current_results)} images")
        
        if self.current_results:
            first_image = list(self.current_results.keys())[0]
            self.display_comparison(first_image, list(self.processor.techniques.keys())[0])
    
    def on_image_select(self, event):
        """When user selects a different image"""
        selected_image = self.image_var.get()
        if selected_image and selected_image in self.current_results:
            self.display_comparison(selected_image, self.tech_var.get())
    
    def on_tech_select(self, event):
        """When user selects a different technique"""
        selected_image = self.image_var.get()
        selected_tech = self.tech_var.get()
        
        if selected_image and selected_tech and selected_image in self.current_results:
            self.display_comparison(selected_image, selected_tech)
    
    def display_image(self, image_path):
        """Display a single image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ✅ PERBAIKI: TWREAD -> IMREAD_GRAYSCALE
        if image is not None:
            self.ax[0].imshow(image, cmap='gray')  # ✅ PERBAIKI: inshow -> imshow
            self.ax[0].set_title("Original Image")
            self.ax[0].axis('off')
            self.canvas.draw()
    
    def display_comparison(self, image_name, technique_name):
        """Display comparison between original and processed"""
        if image_name not in self.current_results:
            return
        
        results = self.current_results[image_name]['results']
        metrics = self.current_results[image_name]['metrics']
        
        # Clear axes
        for ax in self.ax:
            ax.clear()
        
        # Display original
        self.ax[0].imshow(results['original'], cmap='gray')  # ✅ PERBAIKI: inshow -> imshow
        self.ax[0].set_title("Original Image")
        self.ax[0].axis('off')
        
        # Display processed
        if technique_name in results:
            self.ax[1].imshow(results[technique_name], cmap='gray')  # ✅ PERBAIKI: inshow -> imshow
            self.ax[1].set_title(technique_name)
            self.ax[1].axis('off')
            
            # Update metrics
            self.update_metrics_display(metrics['original'], metrics[technique_name], technique_name)
        
        self.canvas.draw()
    
    def update_metrics_display(self, orig_metrics, proc_metrics, technique):
        """Update metrics display"""
        metrics_text = f"TECHNIQUE: {technique}\n\n"
        metrics_text += "ORIGINAL:\n"
        metrics_text += f"  Mean Intensity: {orig_metrics['mean_intensity']:.2f}\n"
        metrics_text += f"  Contrast: {orig_metrics['contrast']:.3f}\n"
        metrics_text += f"  Std Deviation: {orig_metrics['std_intensity']:.2f}\n\n"
        
        metrics_text += "PROCESSED:\n"
        metrics_text += f"  Mean Intensity: {proc_metrics['mean_intensity']:.2f}\n"
        metrics_text += f"  Contrast: {proc_metrics['contrast']:.3f}\n"
        metrics_text += f"  Std Deviation: {proc_metrics['std_intensity']:.2f}\n\n"
        
        metrics_text += "IMPROVEMENT:\n"
        metrics_text += f"  Brightness Δ: {proc_metrics['mean_intensity'] - orig_metrics['mean_intensity']:+.2f}\n"
        metrics_text += f"  Contrast Ratio: {proc_metrics['contrast'] / orig_metrics['contrast']:.3f}x\n"
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
    
    def save_results(self):
        """Save all processed images"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to save! Process images first.")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if output_dir:
            try:
                for image_name, data in self.current_results.items():
                    self.processor.save_results(data['results'], output_dir, image_name)
                
                self.status_var.set(f"Results saved to {output_dir}")
                messagebox.showinfo("Success", "All results saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")