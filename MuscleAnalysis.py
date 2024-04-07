import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk

from gui_inference import *

class ImageInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Inference App")

        # Create widgets
        self.label = tk.Label(root, text="Select an image:")
        self.label.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Orientation option
        self.orientation_var = tk.StringVar(value="None")  # Default to None
        self.orientation_label = tk.Label(root, text="Are the fascicles in the image oriented from bottom-left to top-right?")
        self.orientation_label.pack()
        self.orientation_yes = tk.Radiobutton(root, text="Yes", variable=self.orientation_var, value=0)
        self.orientation_yes.pack()
        self.orientation_no = tk.Radiobutton(root, text="No", variable=self.orientation_var, value=1)
        self.orientation_no.pack()

        # Save directory entry
        self.save_label = tk.Label(root, text="Save inference image to:")
        self.save_label.pack()
        self.select_dir_button = tk.Button(root, text="Select Directory", command=self.select_save_directory)
        self.select_dir_button.pack()
        self.save_dir_label = tk.Label(root, text="")
        self.save_dir_label.pack()
        self.save_dir = None
        self.save_path = None

        # Conversion ratio options
        self.mm_label = tk.Label(root, text="Enter conversion ratio (mm):")
        self.mm_label.pack()
        self.mm_entry = tk.Entry(root)
        self.mm_entry.pack()

        self.pixel_label = tk.Label(root, text="Enter conversion ratio (pixels):")
        self.pixel_label.pack()
        self.pixel_entry = tk.Entry(root)
        self.pixel_entry.pack()

        self.inference_button = tk.Button(root, text="Perform Inference", command=self.perform_inference)
        self.inference_button.pack()

        self.result_label = tk.Label(root)
        self.result_label.pack()



    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            self.image.thumbnail((300, 300))  # Resize image
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)


    def select_save_directory(self):
        self.save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if self.save_path:
            # Extract the directory and filename from the selected path
            directory = os.path.dirname(self.save_path)
            filename = os.path.basename(self.save_path)
            self.save_dir = directory
            
            # Update the save directory label
            self.save_dir_label.config(text="Save As: " + self.save_path)
        

    def perform_inference(self):
        if hasattr(self, 'image'):
            # Check if orientation is selected
            if self.orientation_var.get() == "None":
                self.result_label.config(text="Please select the orientation first.")
                return
            
            # Check if save directory is selected
            if self.save_path == None:
                self.result_label.config(text="Please select the save directory first.")
                return
            
            # Check if mm ratio is specified
            if self.mm_entry.get() == "":
                self.result_label.config(text="Please specify the conversion ratio (mm) first.")
                return
            
            # Check if mm ratio is specified
            if self.pixel_entry.get() == "":
                self.result_label.config(text="Please specify the conversion ratio (pixels) first.")
                return
        
            # Perform inference
            orientation = int(self.orientation_var.get())
            mm_ratio = float(self.mm_entry.get())
            pixel_ratio = float(self.pixel_entry.get())
            result_image_path = infer_image(self.image_path, orientation, mm_ratio, pixel_ratio, self.save_path) 

            # Display the inferred image
            result_image = Image.open(result_image_path)
            # Crop the image to the center
            width, height = result_image.size
            left = (width - height) // 2
            top = (height - height) // 2
            right = (width + height) // 2
            bottom = (height + height) // 2
            result_image = result_image.crop((left, top, right, bottom))
            result_image.thumbnail((300, 300))
            result_photo = ImageTk.PhotoImage(result_image)
            self.result_label.config(image=result_photo)
            self.result_label.image = result_photo  # Keep a reference to prevent garbage collection
        else:
            self.result_label.config(text="Please load an image first.")

def main():
    root = tk.Tk()
    app = ImageInferenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
