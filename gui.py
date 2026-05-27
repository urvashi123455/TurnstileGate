import cv2
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
from datetime import datetime

# ---------------- AUTO DETECT PORTS ----------------
ports = serial.tools.list_ports.comports()
port_list = [p.device for p in ports]

# ---------------- GUI WINDOW ----------------
root = tk.Tk()
root.title("Smart Turnstile Face Recognition")
root.geometry("900x700")
root.configure(bg="white")

# ---------------- TITLE ----------------
title = tk.Label(
    root,
    text="🔐 Turnstile Gate with Face Recognition",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="darkblue"
)
title.pack(pady=10)

# ---------------- COM PORT SELECTION ----------------
port_label = tk.Label(
    root,
    text="Select COM Port:",
    font=("Arial", 12),
    bg="white"
)
port_label.pack()

selected_port = tk.StringVar()

port_dropdown = ttk.Combobox(
    root,
    textvariable=selected_port,
    values=port_list,
    state="readonly",
    width=20
)
port_dropdown.pack(pady=5)

if len(port_list) > 0:
    port_dropdown.current(0)

# ---------------- STATUS ----------------
status_label = tk.Label(
    root,
    text="System Ready",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="green"
)
status_label.pack(pady=10)

# ---------------- LOG BOX ----------------
log_box = tk.Text(root, height=10, width=80)
log_box.pack(pady=10)

# ---------------- CAMERA LABEL ----------------
camera_label = tk.Label(root)
camera_label.pack()

# ---------------- LOAD MODELS ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- CAMERA ----------------
cap = None
running = False

# ---------------- START SYSTEM ----------------
def start_system():

    global cap, running, ser

    if selected_port.get() == "":
        messagebox.showerror(
            "Error",
            "Please Select COM Port"
        )
        return

    try:
        ser = serial.Serial(
            selected_port.get(),
            9600
        )

    except:
        messagebox.showerror(
            "Error",
            "Unable to Open COM Port"
        )
        return

    cap = cv2.VideoCapture(0)

    running = True

    update_frame()

# ---------------- STOP SYSTEM ----------------
def stop_system():

    global running

    running = False

    if cap:
        cap.release()

    cv2.destroyAllWindows()

    status_label.config(
        text="System Stopped",
        fg="red"
    )

# ---------------- UPDATE CAMERA ----------------
last_action_time = 0

def update_frame():

    global last_action_time

    if not running:
        return

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            1.3,
            5
        )

        status = "NO FACE"

        for (x, y, w, h) in faces:

            id, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            current_time = time.time()

            # SMART THRESHOLD
            if confidence < 50:

                status = "AUTHORIZED"

                if current_time - last_action_time > 3:

                    ser.write(b'1')

                    last_action_time = current_time

                    now = datetime.now()

                    log_box.insert(
                        tk.END,
                        f"{now} - AUTHORIZED\n"
                    )

            elif confidence > 80:

                status = "UNAUTHORIZED"

                if current_time - last_action_time > 3:

                    ser.write(b'0')

                    last_action_time = current_time

                    now = datetime.now()

                    log_box.insert(
                        tk.END,
                        f"{now} - UNAUTHORIZED\n"
                    )

            else:
                status = "SUSPICIOUS"

            # Rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                status,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        # STATUS LABEL
        if status == "AUTHORIZED":

            status_label.config(
                text="✅ ACCESS GRANTED",
                fg="green"
            )

        elif status == "UNAUTHORIZED":

            status_label.config(
                text="❌ ACCESS DENIED",
                fg="red"
            )

        else:

            status_label.config(
                text="⚠ VERIFYING...",
                fg="orange"
            )

        # Convert frame
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(rgb)

        img = img.resize((700, 500))

        imgtk = ImageTk.PhotoImage(image=img)

        camera_label.imgtk = imgtk

        camera_label.configure(image=imgtk)

    camera_label.after(10, update_frame)

# ---------------- BUTTONS ----------------
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=10)

start_btn = tk.Button(
    button_frame,
    text="Start System",
    command=start_system,
    font=("Arial", 12, "bold"),
    bg="green",
    fg="white",
    width=15
)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(
    button_frame,
    text="Stop System",
    command=stop_system,
    font=("Arial", 12, "bold"),
    bg="red",
    fg="white",
    width=15
)
stop_btn.grid(row=0, column=1, padx=10)

# ---------------- RUN GUI ----------------
root.mainloop()