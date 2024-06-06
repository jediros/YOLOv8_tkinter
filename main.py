import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from ultralytics import YOLO

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Processing")
        self.root.geometry("300x250")

        self.model = YOLO('best_class.pt')

        self.single_upload_btn = tk.Button(root, text="Select Video", command=self.select_video)
        self.single_upload_btn.pack(pady=5)

        self.directory_upload_btn = tk.Button(root, text="Select Directory", command=self.select_directory)
        self.directory_upload_btn.pack(pady=5)

        self.threshold_label = tk.Label(root, text="Confidence Threshold:")
        self.threshold_label.pack()

        self.threshold_entry = tk.Entry(root)
        self.threshold_entry.pack()

        self.single_process_btn = tk.Button(root, text="Process Single Video", command=self.process_single_video)
        self.single_process_btn.pack(pady=5)

        self.directory_process_btn = tk.Button(root, text="Process Videos in Directory", command=self.process_videos)
        self.directory_process_btn.pack(pady=5)

        self.video_path = ""
        self.video_directory = ""
        self.confidence_threshold = 0.3

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            messagebox.showinfo("Selected Video", f"Selected Video: {self.video_path}")

    def select_directory(self):
        self.video_directory = filedialog.askdirectory()
        if self.video_directory:
            messagebox.showinfo("Selected Directory", f"Selected Directory: {self.video_directory}")

    def process_single_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        self._process_video(self.video_path)

    def process_videos(self):
        if not self.video_directory:
            messagebox.showerror("Error", "Please select a directory containing video files")
            return
        video_files = [f for f in os.listdir(self.video_directory) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            messagebox.showerror("Error", "No video files found in the selected directory")
            return
        for video_file in video_files:
            video_path = os.path.join(self.video_directory, video_file)
            self._process_video(video_path)
        messagebox.showinfo("Done", "Video processing complete. Output saved in the selected directory")

    def _process_video(self, video_path):
        try:
            self.confidence_threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid confidence threshold")
            return

        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.splitext(video_path)[0] + '_output.avi', fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                results = self.model.track(frame, conf=self.confidence_threshold, persist=True)
                frame_ = results[0].plot()

                # Resize the frame by a factor of 2
                frame_resized = cv2.resize(frame_, (frame_.shape[1] // 3, frame_.shape[0] // 3))

                out.write(frame_)
                cv2.imshow('frame', frame_resized)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Video processing complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
