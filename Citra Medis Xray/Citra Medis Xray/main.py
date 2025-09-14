import tkinter as tk
from modules.image_processor import ImageProcessor
from modules.gui import MedicalImageGUI

def main():
    """Main function to start the application"""
    # Initialize processor
    processor = ImageProcessor()
    
    # Create GUI
    root = tk.Tk()
    app = MedicalImageGUI(root, processor)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()