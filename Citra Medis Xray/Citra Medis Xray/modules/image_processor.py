import cv2
import numpy as np
from pathlib import Path

class ImageProcessor:
    def __init__(self):
        self.techniques = {
            'Histogram Equalization': self.apply_histogram_equalization,
            'CLAHE (clip=2.0, grid=8x8)': lambda img: self.apply_clahe(img, 2.0, (8, 8)),
            'CLAHE (clip=4.0, grid=16x16)': lambda img: self.apply_clahe(img, 4.0, (16, 16)),
            'Gamma Correction (γ=0.5)': lambda img: self.adjust_gamma(img, 0.5),
            'Gamma Correction (γ=1.5)': lambda img: self.adjust_gamma(img, 1.5),
            'Gaussian Filter (5x5)': self.apply_gaussian_filter,
            'Median Filter (5x5)': self.apply_median_filter
        }
    
    def adjust_gamma(self, image, gamma=1.0):
        """Adjust image gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def apply_histogram_equalization(self, image):
        """Apply histogram equalization"""
        return cv2.equalizeHist(image)
    
    def apply_clahe(self, image, clip_limit=2.0, grid_size=(8,8)):
        """Apply CLAHE with specified parameters"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)
    
    def apply_gaussian_filter(self, image):
        """Apply Gaussian filter"""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def apply_median_filter(self, image):
        """Apply Median filter"""
        return cv2.medianBlur(image, 5)
    
    def calculate_metrics(self, image):
        """Calculate image quality metrics"""
        if image is None:
            return {}
        
        return {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'min_intensity': np.min(image),
            'max_intensity': np.max(image),
            'contrast': np.std(image) / np.mean(image) if np.mean(image) > 0 else 0
        }
    
    def process_single_image(self, image_path):
        """Process a single image with all techniques"""
        if not Path(image_path).exists():
            return None, None
        
        # Read image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            return None, None
        
        results = {'original': original}
        metrics = {'original': self.calculate_metrics(original)}
        
        # Apply all techniques
        for technique_name, technique_func in self.techniques.items():
            processed_image = technique_func(original)
            results[technique_name] = processed_image
            metrics[technique_name] = self.calculate_metrics(processed_image)
        
        return results, metrics
    
    def process_batch(self, image_paths):
        """Process multiple images"""
        all_results = {}
        
        for image_path in image_paths:
            image_name = Path(image_path).name
            results, metrics = self.process_single_image(image_path)
            
            if results:
                all_results[image_name] = {
                    'results': results,
                    'metrics': metrics,
                    'path': image_path
                }
        
        return all_results
    
    def save_results(self, results, output_dir, image_name):
        """Save processed images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for technique, image in results.items():
            # Clean filename
            clean_technique = "".join(c if c.isalnum() else "_" for c in technique)
            filename = f"{image_name}_{clean_technique}.png"
            cv2.imwrite(str(output_dir / filename), image)
        
        return True