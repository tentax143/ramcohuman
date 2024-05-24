import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import os
import multiprocessing as mp

def process_video(frame_queue, result_queue, yolo_model_path, video_source, reg_pts):
    if not os.path.exists(yolo_model_path):
        print("Downloading YOLOv8 model file...")
        YOLO(yolo_model_path)  # This will download the file if not present

    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), "Error reading video file"

    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=reg_pts,
                     classes_names=model.names,
                     draw_tracks=True,
                     line_thickness=2)

    while True:
        success, frame = cap.read()
        if not success:
            break

        tracks = model.track(frame, persist=True, show=False, classes=[0, 2])
        processed_frame = counter.start_counting(frame, tracks)

        result = {
            "in_count": counter.in_counts,
            "out_count": counter.out_counts,
            "car_count": counter.in_counts,
            "frame": processed_frame
        }

        if not frame_queue.empty():
            _ = frame_queue.get()  # Discard the previous frame request

        result_queue.put(result)

    cap.release()

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Human Detector")
        self.geometry("1942x978")

        self.create_widgets()

        self.frame_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.process = mp.Process(target=process_video,
                                  args=(self.frame_queue, self.result_queue,
                                        "yolov8n.pt", "rtsp://admin:admin123%23@172.16.3.21:554",
                                        [(260, 300), (1100, 250)]))
        self.process.start()

        self.in_count = 0
        self.out_count = 0
        self.inside_count = 0
        self.car_count_val = 0

        self.update_frame()

    def create_widgets(self):
        self.font = tkfont.Font(family='Poppins ExtraBold', size=14, weight='bold')

        self.video_frame = tk.Label(self)
        self.video_frame.place(x=40, y=40, width=1200, height=900)

        x_offset = 1300
        y_offset = 100
        y_space = 70

        self.total_in_label = tk.Label(self, text="TOTAL ENTRY", font=self.font)
        self.total_in_label.place(x=x_offset, y=y_offset)

        self.total_in_display = tk.Label(self, font=self.font)
        self.total_in_display.place(x=x_offset, y=y_offset + y_space)
        self.total_in_display.config(fg='green')

        self.total_out_label = tk.Label(self, text="TOTAL EXIT", font=self.font)
        self.total_out_label.place(x=x_offset, y=y_offset + y_space * 2)

        self.total_out_display = tk.Label(self, font=self.font)
        self.total_out_display.place(x=x_offset, y=y_offset + y_space * 3)
        self.total_out_display.config(fg='red')

        self.total_inside_label = tk.Label(self, text="TOTAL INSIDE", font=self.font)
        self.total_inside_label.place(x=x_offset, y=y_offset + y_space * 4)

        self.total_inside_display = tk.Label(self, font=self.font)
        self.total_inside_display.place(x=x_offset, y=y_offset + y_space * 5)
        self.total_inside_display.config(fg='#E3651D')

        self.car_count_label = tk.Label(self, text="TRANSPORT ENTRY", font=self.font)
        self.car_count_label.place(x=x_offset, y=y_offset + y_space * 6)

        self.car_count_display = tk.Label(self, font=self.font)
        self.car_count_display.place(x=x_offset, y=y_offset + y_space * 7)
        self.car_count_display.config(fg='blue')

        self.total_transport_entry_label = tk.Label(self, text="TOTAL IN TRANSPORT", font=self.font)
        self.total_transport_entry_label.place(x=x_offset, y=y_offset + y_space * 8)

        self.total_transport_entry = tk.Entry(self, font=self.font)
        self.total_transport_entry.place(x=x_offset, y=y_offset + y_space * 9)
        self.total_transport_entry.bind('<Return>', self.update_transport_entry)

        self.total_out_transport_label = tk.Label(self, text="TOTAL OUT TRANSPORT", font=self.font)
        self.total_out_transport_label.place(x=x_offset, y=y_offset + y_space * 10)

        self.total_out_transport_entry = tk.Entry(self, font=self.font)
        self.total_out_transport_entry.place(x=x_offset, y=y_offset + y_space * 11)

    def update_transport_entry(self, event=None):
        try:
            value = int(self.total_transport_entry.get().strip())
            self.in_count += value
            self.total_transport_entry.delete(0, tk.END)
            self.total_in_display.config(text=str(self.in_count))
            self.total_inside_display.config(text=str(self.in_count - self.out_count))
        except ValueError:
            pass

    def update_frame(self):
        if not self.result_queue.empty():
            result = self.result_queue.get()

            current_in_count = result['in_count']
            current_out_count = result['out_count']
            current_car_count = result['car_count']
            im0 = result['frame']

            if current_out_count > self.out_count:
                self.out_count = current_out_count

            if current_in_count > self.in_count:
                self.in_count = current_in_count

            if self.out_count < 0:
                self.out_count = 0
                print("Out count adjusted to 0 to prevent negative value.")

            self.inside_count = self.in_count - self.out_count

            self.total_in_display.config(text=str(self.in_count))
            self.total_out_display.config(text=str(self.out_count))
            self.total_inside_display.config(text=str(self.inside_count))

            if current_car_count != self.car_count_val:
                self.car_count_val = current_car_count
                self.car_count_display.config(text=str(self.car_count_val))

            rgb_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            image.thumbnail((1000, 800), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.video_frame.config(image=photo)
            self.video_frame.image = photo

        self.after(30, self.update_frame)

    def on_closing(self):
        self.process.terminate()
        self.destroy()

if __name__ == "__main__":
    mp.freeze_support()
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
