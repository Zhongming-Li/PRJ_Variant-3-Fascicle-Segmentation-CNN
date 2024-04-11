import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from gui_inference import *

class ImageInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Inference App")

        # Create widgets
        self.label = tk.Label(root, text="Select an image:")
        self.label.grid(row=0, column=0, sticky=tk.W)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.grid(row=1, column=0, sticky=tk.W)

        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=1, rowspan=12, columnspan=1, sticky=tk.W)

        # Orientation option
        self.orientation_label = tk.Label(root, text="Are the fascicles in the image oriented from bottom-left to top-right?")
        self.orientation_label.grid(row=3, column=0, sticky=tk.W)

        self.orientation_var = tk.StringVar(value="None")
        self.orientation_yes = tk.Radiobutton(root, text="Yes", variable=self.orientation_var, value=0)
        self.orientation_yes.grid(row=4, column=0, sticky=tk.W)

        self.orientation_no = tk.Radiobutton(root, text="No", variable=self.orientation_var, value=1)
        self.orientation_no.grid(row=5, column=0, sticky=tk.W)

        # Save directory entry
        self.save_label = tk.Label(root, text="Save inference image to:")
        self.save_label.grid(row=6, column=0, sticky=tk.W)

        self.select_dir_button = tk.Button(root, text="Select Directory", command=self.select_save_directory)
        self.select_dir_button.grid(row=7, column=0, sticky=tk.W)

        self.save_dir_label = tk.Label(root, text="")
        self.save_dir_label.grid(row=8, column=0, sticky=tk.W)

        # Conversion ratio options
        self.mm_label = tk.Label(root, text="Enter conversion ratio (mm):")
        self.mm_label.grid(row=9, column=0, sticky=tk.W)

        self.mm_entry = tk.Entry(root)
        self.mm_entry.grid(row=10, column=0, sticky=tk.W)

        self.pixel_label = tk.Label(root, text="Enter conversion ratio (pixels):")
        self.pixel_label.grid(row=11, column=0, sticky=tk.W)

        self.pixel_entry = tk.Entry(root)
        self.pixel_entry.grid(row=12, column=0, sticky=tk.W)

        self.inference_button = tk.Button(root, text="Perform Inference", command=self.perform_inference)
        self.inference_button.grid(row=13, column=0, sticky=tk.W)

        self.result_label = tk.Label(root)
        self.result_label.grid(row=0, column=1, rowspan=12, columnspan=1, sticky=tk.W)

        self.error_label = tk.Label(root)
        self.error_label.grid(row=13, column=1, columnspan=2, sticky=tk.W)

        # Clear All button
        self.clear_button = tk.Button(root, text="Clear All", command=self.clear_all)
        self.clear_button.grid(row=14, column=0, sticky=tk.W)


    def load_image(self):
        file_path = filedialog.askopenfilename()
        self.image_path = file_path
        print(f"==========self image path: {file_path}")
        input_image = Image.open(file_path)
        input_image.thumbnail((300, 300))
        input_photo = ImageTk.PhotoImage(input_image)
        self.image_label.config(image=input_photo)
        self.image_label.image = input_photo
        print("=========load image")


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

        # Check if original image is uploaded
        if not hasattr(self, 'image_path'):
            self.error_label.config(text="Please load an image first.")
            return
        elif self.image_path=="":
            self.error_label.config(text="Please load an image first.")
            return

        # Check if orientation is selected
        if self.orientation_var.get() == "None":
            self.error_label.config(text="Please select the orientation first.")
            return
        
        # Check if save directory is selected
        if self.save_path == None:
            self.error_label.config(text="Please select the save directory first.")
            return
        
        # Check if mm ratio is specified
        if self.mm_entry.get() == "":
            self.error_label.config(text="Please specify the conversion ratio (mm) first.")
            return
        # Check if mm ratio is valid
        try:
            float(self.mm_entry.get())
        except ValueError:
            self.error_label.config(text="Please enter the conversion ratio (mm) as integers or floats.")
            return
        
        # Check if pixel ratio is specified
        if self.pixel_entry.get() == "":
            self.error_label.config(text="Please specify the conversion ratio (pixels) first.")
            return
        # Check if pixel ratio is valid
        try:
            float(self.pixel_entry.get())
        except ValueError:
            self.error_label.config(text="Please enter the conversion ratio (pixels) as integers or floats.")
            return
        
        
        
        # Perform inference
        orientation = int(self.orientation_var.get())
        mm_ratio = float(self.mm_entry.get())
        pixel_ratio = float(self.pixel_entry.get())
        self.error_label.config(text="Analyzing image...")
        result_image_path = infer_image(self.image_path, orientation, mm_ratio, pixel_ratio, self.save_path) 

        # Display the inferred image
        result_image = Image.open(result_image_path)
        width, height = result_image.size
        left = (width - height) // 2
        top = (height - height) // 2
        right = (width + height) // 2
        bottom = (height + height) // 2
        result_image = result_image.crop((left, top, right, bottom))
        result_image.thumbnail((300, 300))
        result_photo = ImageTk.PhotoImage(result_image)                
        self.image_label.config(image=result_photo)
        self.image_label.image = result_photo

        self.error_label.config(text="")
        

    def clear_all(self):
        # Reset all input fields to their default state or clear their content
        self.image_path = ""
        self.image_label.config(image=None)
        self.image_label.image = None

        self.orientation_var.set("None")
        self.mm_entry.delete(0, tk.END)
        self.pixel_entry.delete(0, tk.END)
        self.save_dir_label.config(text="")

        self.result_label.config(image=None) 
        self.result_label.image = None

        self.error_label.config(text="")




def main():
    root = tk.Tk()
    app = ImageInferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", on_close(root))
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()

    def on_close():
        # End the application when the window is closed
        root.quit()
        root.destroy()

    app = ImageInferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    # main()
