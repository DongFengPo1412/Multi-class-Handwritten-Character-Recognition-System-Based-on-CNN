import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.baidu_ocr import BaiduOCRClient
from src.local_ocr import LocalOCRRecognizer


COLORS = {
    "background": "#f3f5f7",
    "surface": "#ffffff",
    "border": "#dfe3e8",
    "text": "#252a31",
    "muted": "#7c838d",
    "green": "#168a61",
    "orange": "#d97706",
    "blue": "#2878c8",
    "dark": "#20242a",
    "warning_bg": "#fff8e7",
    "warning_border": "#efd797",
}


def cv_to_photo(image, max_width, max_height):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(max_width / width, max_height / height)
    resized = cv2.resize(
        rgb,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return ImageTk.PhotoImage(Image.fromarray(resized))


def cv_to_preview_photo(image, width, height):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
    return ImageTk.PhotoImage(Image.fromarray(resized))


class ResultCard(tk.Frame):
    def __init__(self, parent, title, color):
        super().__init__(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=18,
            pady=13,
        )
        tk.Label(
            self,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x")
        self.value = tk.Label(
            self,
            text="--",
            bg=COLORS["surface"],
            fg=color,
            font=("Segoe UI", 20, "bold"),
            anchor="w",
            justify="left",
            wraplength=380,
        )
        self.value.pack(fill="x", pady=(4, 1))
        self.detail = tk.Label(
            self,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=380,
        )
        self.detail.pack(fill="x")

    def set_result(self, value, detail=""):
        self.value.configure(text=value or "--")
        self.detail.configure(text=detail)


class OCRDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten OCR Studio")
        self.root.geometry("1440x850")
        self.root.minsize(1120, 700)
        self.root.configure(bg=COLORS["background"])

        self.recognizer = LocalOCRRecognizer()
        self.baidu_client = None
        self.baidu_enabled = True
        self.frozen = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.current_frame = None
        self.current_roi = None
        self.video_photo = None
        self.preview_photo = None
        self.preview_source = None
        self.preview_resize_job = None
        self.log_entries = []
        self.log_window = None
        self.log_text = None

        self.camera_indices = self.scan_cameras()
        self.camera_position = 0
        self.capture = self.open_camera(self.camera_indices[0])

        self.build_ui()
        self.bind_keys()
        self.initialize_baidu()
        self.update_camera()
        self.poll_result()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    @staticmethod
    def scan_cameras(max_index=4):
        indices = []
        for index in range(max_index):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    indices.append(index)
            capture.release()
        return indices or [0]

    @staticmethod
    def open_camera(index):
        capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return capture

    def build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["background"])
        header.pack(fill="x", padx=24, pady=(18, 14))
        title_box = tk.Frame(header, bg=COLORS["background"])
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="Handwritten OCR Studio",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="CNN local recognition with Baidu handwriting reference",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        self.status_badge = tk.Label(
            header,
            text="LIVE",
            bg=COLORS["green"],
            fg="white",
            padx=12,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_badge.pack(side="right", padx=(8, 0))
        self.baidu_badge = tk.Label(
            header,
            text="BAIDU CONNECTING",
            bg="#7d848d",
            fg="white",
            padx=12,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.baidu_badge.pack(side="right", padx=(8, 0))
        self.camera_badge = tk.Label(
            header,
            text=f"CAM {self.camera_indices[0]}",
            bg=COLORS["blue"],
            fg="white",
            padx=12,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.camera_badge.pack(side="right")

        body = tk.Frame(self.root, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        camera_panel = tk.Frame(
            body,
            bg="#17191d",
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.video_label = tk.Label(camera_panel, bg="#17191d")
        self.video_label.pack(fill="both", expand=True)

        side = tk.Frame(body, bg=COLORS["background"], width=430)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        buttons = tk.Frame(side, bg=COLORS["background"])
        buttons.pack(side="bottom", fill="x")
        self.recognize_button = tk.Button(
            buttons,
            text="Recognize",
            command=self.start_recognition,
            bg=COLORS["green"],
            fg="white",
            activebackground="#117653",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=9,
            font=("Segoe UI", 9, "bold"),
        )
        self.recognize_button.pack(side="left")
        tk.Button(
            buttons,
            text="Switch Camera",
            command=self.switch_camera,
            relief="solid",
            bd=1,
            padx=12,
            pady=8,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=8)
        self.baidu_button = tk.Button(
            buttons,
            text="Baidu: On",
            command=self.toggle_baidu,
            relief="solid",
            bd=1,
            padx=12,
            pady=8,
            font=("Segoe UI", 9),
        )
        self.baidu_button.pack(side="left")

        self.raw_card = ResultCard(side, "原始识别结果 (LOCAL CNN / RAW)", COLORS["orange"])
        self.raw_card.pack(fill="x", pady=(0, 10))
        self.corrected_card = ResultCard(side, "推理校正结果 (CORRECTED)", COLORS["green"])
        self.corrected_card.pack(fill="x", pady=(0, 10))
        self.baidu_card = ResultCard(side, "百度云识别结果 (BAIDU OCR)", COLORS["blue"])
        self.baidu_card.pack(fill="x", pady=(0, 10))

        preview_card = tk.Frame(
            side,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=18,
            pady=13,
        )
        preview_card.pack(fill="x", pady=(0, 10))
        tk.Label(
            preview_card,
            text="实时二值化动态跟踪图 (AI VISION)",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        preview_view = tk.Frame(
            preview_card,
            bg="#fafafa",
            height=210,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        preview_view.pack(fill="x", pady=(8, 0))
        preview_view.pack_propagate(False)
        self.preview_label = tk.Label(preview_view, bg="#fafafa")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_label.bind("<Configure>", self.schedule_preview_render)

        notice_panel = tk.Frame(
            side,
            bg=COLORS["warning_bg"],
            highlightbackground=COLORS["warning_border"],
            highlightthickness=1,
        )
        notice_panel.pack(fill="x", pady=(0, 10))
        notice_header = tk.Frame(notice_panel, bg=COLORS["warning_bg"])
        notice_header.pack(fill="x", padx=12, pady=(7, 0))
        tk.Label(
            notice_header,
            text="LOG",
            bg=COLORS["warning_bg"],
            fg="#805c17",
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")
        tk.Button(
            notice_header,
            text="Expand",
            command=self.open_log_window,
            bg=COLORS["warning_bg"],
            fg="#805c17",
            activebackground="#f8edcf",
            activeforeground="#805c17",
            relief="flat",
            bd=0,
            padx=2,
            pady=0,
            cursor="hand2",
            font=("Segoe UI", 8, "underline"),
        ).pack(side="right")
        self.notice = tk.Label(
            notice_panel,
            text="Ready. Place handwriting inside the capture area.",
            bg=COLORS["warning_bg"],
            fg="#805c17",
            padx=12,
            pady=7,
            anchor="w",
            justify="left",
            height=1,
            font=("Segoe UI", 9),
        )
        self.notice.pack(fill="x")
        self.add_log("Ready. Place handwriting inside the capture area.")

        footer = tk.Label(
            self.root,
            text="SPACE  Recognize/Resume     B  Baidu OCR     C  Camera     ENTER/ESC  Resume     Q  Quit",
            bg=COLORS["surface"],
            fg="#626a74",
            anchor="w",
            padx=24,
            pady=9,
            font=("Segoe UI", 9),
        )
        footer.pack(fill="x")

    def bind_keys(self):
        self.root.bind("<space>", lambda _event: self.start_recognition())
        self.root.bind("b", lambda _event: self.toggle_baidu())
        self.root.bind("c", lambda _event: self.switch_camera())
        self.root.bind("<Return>", lambda _event: self.unfreeze() if self.frozen else None)
        self.root.bind("<Escape>", lambda _event: self.unfreeze() if self.frozen else None)
        self.root.bind("q", lambda _event: self.close())

    def add_log(self, message, summary=None):
        self.log_entries.append(message)
        self.log_entries = self.log_entries[-100:]
        compact = " ".join((summary or message).split())
        if len(compact) > 74:
            compact = f"{compact[:71]}..."
        self.notice.configure(text=compact)
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", "\n".join(self.log_entries))
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def schedule_preview_render(self, _event=None):
        if self.preview_source is None:
            return
        if self.preview_resize_job is not None:
            self.root.after_cancel(self.preview_resize_job)
        self.preview_resize_job = self.root.after(80, self.render_preview)

    def render_preview(self):
        self.preview_resize_job = None
        if self.preview_source is None:
            return
        width = max(1, self.preview_label.winfo_width())
        height = max(1, self.preview_label.winfo_height())
        self.preview_photo = cv_to_preview_photo(self.preview_source, width, height)
        self.preview_label.configure(image=self.preview_photo)

    def open_log_window(self):
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
            self.log_window.focus_force()
            return

        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Recognition Log")
        self.log_window.geometry("720x380")
        self.log_window.minsize(520, 260)
        self.log_window.configure(bg=COLORS["background"])

        container = tk.Frame(self.log_window, bg=COLORS["surface"], padx=14, pady=14)
        container.pack(fill="both", expand=True, padx=14, pady=14)
        scrollbar = tk.Scrollbar(container)
        scrollbar.pack(side="right", fill="y")
        self.log_text = tk.Text(
            container,
            wrap="word",
            yscrollcommand=scrollbar.set,
            bg="#fffdf7",
            fg=COLORS["text"],
            relief="flat",
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.log_text.pack(fill="both", expand=True)
        scrollbar.configure(command=self.log_text.yview)
        self.log_text.insert("end", "\n".join(self.log_entries))
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def initialize_baidu(self):
        try:
            self.baidu_client = BaiduOCRClient()
            self.baidu_client.ensure_ready()
            self.baidu_badge.configure(text="BAIDU READY", bg="#24965b")
        except Exception as exc:
            self.baidu_enabled = False
            self.baidu_badge.configure(text="BAIDU OFFLINE", bg="#7d848d")
            self.baidu_button.configure(text="Baidu: Off")
            self.baidu_card.set_result("--", str(exc))

    def update_camera(self):
        if not self.frozen:
            ok, frame = self.capture.read()
            if ok:
                self.current_frame = frame
                height, width = frame.shape[:2]
                size = min(420, int(min(height, width) * 0.62))
                x1 = (width - size) // 2
                y1 = (height - size) // 2
                x2, y2 = x1 + size, y1 + size
                self.current_roi = frame[y1:y2, x1:x2].copy()

                dimmed = cv2.addWeighted(frame, 0.48, np.full_like(frame, 15), 0.52, 0)
                dimmed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                color = (196, 154, 28)
                length = 42
                for start, end in [
                    ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
                    ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
                    ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
                    ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
                ]:
                    cv2.line(dimmed, start, end, color, 3, cv2.LINE_AA)

                panel_width = max(self.video_label.winfo_width(), 700)
                panel_height = max(self.video_label.winfo_height(), 500)
                self.video_photo = cv_to_photo(dimmed, panel_width, panel_height)
                self.video_label.configure(image=self.video_photo)

                # Real-time binarization tracking preview
                if self.current_roi is not None:
                    roi = self.current_roi
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    bg = cv2.GaussianBlur(gray, (51, 51), 0)
                    gray_no_shadow = cv2.divide(gray, bg, scale=255)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray_no_shadow)
                    blur = cv2.GaussianBlur(cv2.medianBlur(enhanced, 5), (3, 3), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 5)
                    h_t, w_t = thresh.shape
                    border_pixels = np.concatenate([
                        thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]
                    ])
                    if np.mean(border_pixels) > 127:
                        thresh = 255 - thresh

                    self.preview_source = 255 - thresh
                    self.render_preview()
            else:
                warning_msg = "[Warning] Failed to read frame from camera. Check camera connection or index."
                print(warning_msg)
                self.notice.configure(text=warning_msg)
                
        self.root.after(30, self.update_camera)

    def start_recognition(self):
        if self.frozen:
            self.unfreeze()
            return

        if self.current_roi is None or self.pending is not None:
            return
        
        # Freeze camera and binary previews
        self.frozen = True
        self.recognize_button.configure(state="disabled", text="Recognizing...")
        self.status_badge.configure(text="FREEZE", bg=COLORS["orange"])
        self.add_log("画面已定格。正在运行推理识别与纠错处理...")
        self.pending = self.executor.submit(self.run_recognition, self.current_roi.copy())

    def run_recognition(self, roi):
        local = self.recognizer.recognize(roi)
        baidu = None
        error = ""
        if self.baidu_enabled and self.baidu_client is not None:
            try:
                baidu = self.baidu_client.recognize_ndarray(roi)
            except Exception as exc:
                error = str(exc)
        return local, baidu, error

    def poll_result(self):
        if self.pending is not None and self.pending.done():
            future = self.pending
            self.pending = None
            # Stay in frozen state, recognize_button text becomes "Resume" (unfreeze trigger)
            self.recognize_button.configure(state="normal", text="Resume")
            try:
                local, baidu, error = future.result()
                self.show_result(local, baidu, error)
            except Exception as exc:
                self.add_log(f"Recognition failed: {exc}")
                self.unfreeze()
        self.root.after(80, self.poll_result)

    def show_result(self, local, baidu, error):
        self.raw_card.set_result(
            local.raw_text,
            f"{local.character_count} 个字符 / {local.line_count} 行",
        )
        context = " / ".join(local.contexts) if local.contexts else "neutral"
        self.corrected_card.set_result(local.corrected_text, f"推断语境: {context}")

        self.preview_source = 255 - local.threshold
        self.render_preview()

        # Draw character bounding boxes on the frozen camera frame!
        if self.current_frame is not None and local.boxes:
            marked_frame = self.current_frame.copy()
            height, width = marked_frame.shape[:2]
            size = min(420, int(min(height, width) * 0.62))
            x1 = (width - size) // 2
            y1 = (height - size) // 2
            x2, y2 = x1 + size, y1 + size
            
            # Dim background
            dimmed = cv2.addWeighted(marked_frame, 0.48, np.full_like(marked_frame, 15), 0.52, 0)
            dimmed[y1:y2, x1:x2] = marked_frame[y1:y2, x1:x2]
            
            # Draw bounding boxes and index numbers
            color = (255, 255, 0) # Cyan in BGR
            for idx, (bx, by, bw, bh) in enumerate(local.boxes, 1):
                abs_x1, abs_y1 = x1 + bx, y1 + by
                abs_x2, abs_y2 = abs_x1 + bw, abs_y1 + bh
                cv2.rectangle(dimmed, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)
                cv2.putText(dimmed, str(idx), (abs_x1, max(abs_y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
            # Draw corner overlays
            length = 42
            corner_color = (196, 154, 28)
            for start, end in [
                ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
                ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
                ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
                ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
            ]:
                cv2.line(dimmed, start, end, corner_color, 3, cv2.LINE_AA)

            panel_width = max(self.video_label.winfo_width(), 700)
            panel_height = max(self.video_label.winfo_height(), 500)
            self.video_photo = cv_to_photo(dimmed, panel_width, panel_height)
            self.video_label.configure(image=self.video_photo)

        if baidu is not None:
            confidence = baidu.average_confidence
            detail = f"平均置信度: {confidence:.1%}" if confidence is not None else ""
            self.baidu_card.set_result(baidu.text, detail)
        elif error:
            self.baidu_card.set_result("--", error)
        else:
            self.baidu_card.set_result("已禁用", "")

        if local.uncertain:
            details = " | ".join(local.uncertain)
            self.add_log(
                f"{len(local.uncertain)} 个不确定字符. {details}",
                f"{len(local.uncertain)} 个不确定字符. 点击展开查看详情.",
            )
        else:
            self.add_log("识别完成。所有字符均高置信度通过。")

    def unfreeze(self):
        self.frozen = False
        self.recognize_button.configure(text="Recognize")
        self.status_badge.configure(text="LIVE", bg=COLORS["green"])
        self.add_log("定格已解除，恢复实时相机画面与二值化跟踪。")

    def toggle_baidu(self):
        self.baidu_enabled = not self.baidu_enabled
        self.baidu_button.configure(text=f"Baidu: {'On' if self.baidu_enabled else 'Off'}")
        self.baidu_badge.configure(
            text=f"BAIDU {'ON' if self.baidu_enabled else 'OFF'}",
            bg="#24965b" if self.baidu_enabled else "#7d848d",
        )

    def switch_camera(self):
        self.capture.release()
        self.camera_position = (self.camera_position + 1) % len(self.camera_indices)
        index = self.camera_indices[self.camera_position]
        self.capture = self.open_camera(index)
        self.camera_badge.configure(text=f"CAM {index}")

    def close(self):
        self.capture.release()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    OCRDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
