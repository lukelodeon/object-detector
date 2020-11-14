from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from detector import Detector
import tensorflow_hub as hub

module_handlers = ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1",
                   "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

class GUI:
    def __init__(self):
        root = Tk()
        root.geometry("550x300+300+150")
        root.resizable(width=True, height=True)
        self.root = root
        self.btn = Button(root, text='Open Image', command=self.open_img).pack()
        self.display = None
        self.obj = Detector()
        module_handle = module_handlers[0]
        self.detector = hub.load(module_handle).signatures['default']

    def run(self):
        self.root.mainloop()

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img(self):
        filename = self.openfn()
        img = Image.open(filename)
        img = img.resize((400, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.root, image=img)
        panel.image = img
        panel.pack()
        if self.display == None:
            dis_obj = panel
            self.display = dis_obj
        else:
            self.display.destroy()
            dis_obj = panel
            self.display = dis_obj

        self.display_detect(filename)

    def display_detect(self, filename):
        btn_detect = Button(self.root, text="Run Detection", command=lambda: self.run_detection(filename))
        btn_detect.pack()
        self.btn_detect = btn_detect

    def run_detection(self, filename):
        downloaded_image_path = self.obj.download_and_resize_image(filename, 1280, 856)
        out_filename = self.obj.run_detector(self.detector, downloaded_image_path)
        self.open_out_img(out_filename)

    def open_out_img(self, out_filename):
        img = Image.open(out_filename)
        img = img.resize((400, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.root, image=img)
        panel.image = img
        panel.pack()
        self.display.destroy()
        self.btn_detect.destroy()
        self.display = panel