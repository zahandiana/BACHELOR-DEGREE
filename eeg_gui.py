import tkinter as tk
from tkinter import ttk
from pylsl import StreamInlet, resolve_stream
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import welch

class EEGStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Live Stream")

        # Frame for controls
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Start and stop buttons
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_streaming)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_streaming)
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)

        # Split the screen into two frames
        main_frame = ttk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame for the table and additional visualization
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Frame for the table
        table_frame = ttk.Frame(left_frame)
        table_frame.pack(side=tk.TOP, fill=tk.X)

        # Create Treeview for the table
        self.tree = ttk.Treeview(table_frame, columns=('Channel', 'Value'), show='headings', height=16)
        self.tree.heading('Channel', text='Channel')
        self.tree.heading('Value', text='Value')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the table
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame for additional visualization
        self.bar_frame = ttk.Frame(left_frame)
        self.bar_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for fatigue level
        self.fatigue_canvas = tk.Canvas(self.bar_frame, width=100, height=300, bg='white')
        self.fatigue_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.fatigue_bar = self.fatigue_canvas.create_rectangle(10, 10, 90, 290, fill="blue")
        self.fatigue_label = ttk.Label(self.bar_frame, text="Fatigue Level")
        self.fatigue_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Canvas for focus level
        self.focus_canvas = tk.Canvas(self.bar_frame, width=100, height=300, bg='white')
        self.focus_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.focus_bar = self.focus_canvas.create_rectangle(10, 10, 90, 290, fill="green")
        self.focus_label = ttk.Label(self.bar_frame, text="Focus Level")
        self.focus_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame for the plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Matplotlib figure and axis for the plot
        self.fig, self.ax = plt.subplots(8, 2, figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lines = []
        colors = plt.cm.viridis(np.linspace(0, 1, 16))  # Generate a list of colors
        for i in range(16):
            ax = self.ax[i // 2, i % 2]
            line, = ax.plot([], [], lw=2, color=colors[i], label=f'Channel {i+1}')
            self.lines.append(line)
            ax.set_ylim(-1000, 1000)
            ax.set_xlim(0, 250)
            ax.legend(loc='upper right')
            ax.axhline(y=800, color='r', linestyle='--')  # Threshold line for critical level

        self.xdata = []
        self.ydata = [[] for _ in range(16)]

        self.ani = None

        self.inlet = None
        self.streaming_thread = None
        self.running = False

    def start_streaming(self):
        if self.streaming_thread is None:
            print("Starting streaming thread...")
            self.running = True
            self.streaming_thread = threading.Thread(target=self.stream_data)
            self.streaming_thread.start()
            print("Starting animation...")
            self.ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=50)
            self.canvas.draw()

    def stop_streaming(self):
        print("Stopping streaming...")
        self.running = False
        if self.streaming_thread:
            self.streaming_thread.join()
            self.streaming_thread = None
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        print("Streaming stopped.")

    def stream_data(self):
        print("Resolving EEG stream...")
        streams = resolve_stream('type', 'EEG')
        if streams:
            self.inlet = StreamInlet(streams[0])
            print("Stream resolved.")
        else:
            print("No EEG stream found.")
            return
        
        while self.running:
            sample, timestamp = self.inlet.pull_sample()
            if sample:
                print(f"Sample: {sample}, Timestamp: {timestamp}")
                self.update_table(sample)
                self.update_graph(sample, timestamp)
                self.update_bars(sample)
                self.root.update_idletasks()
            else:
                print("No sample received.")
                self.running = False

    def update_table(self, sample):
        print(f"Updating table with sample: {sample}")
        for i, value in enumerate(sample):
            channel = f'Channel {i+1}'
            formatted_value = f"{value:.2f}"
            if len(self.tree.get_children()) < 16:
                self.tree.insert('', 'end', values=(channel, formatted_value))
            else:
                self.tree.set(self.tree.get_children()[i], column='Value', value=formatted_value)

    def update_graph(self, sample, timestamp):
        print(f"Updating graph with sample: {sample}, Timestamp: {timestamp}")
        self.xdata.append(timestamp)
        for i in range(16):
            self.ydata[i].append(sample[i])
            if len(self.ydata[i]) > 250:  # keep the last 250 points
                self.ydata[i] = self.ydata[i][-250:]
        if len(self.xdata) > 250:  # keep the last 250 points
            self.xdata = self.xdata[-250:]

    def update_plot(self, frame):
        if self.xdata:
            print("Updating plot...")
            for i, line in enumerate(self.lines):
                line.set_data(self.xdata, self.ydata[i])
                ax = self.ax[i // 2, i % 2]
                ax.set_xlim(self.xdata[0], self.xdata[-1])
                ax.set_ylim(min(min(y) for y in self.ydata), max(max(y) for y in self.ydata))
            self.canvas.draw()
        return self.lines

    def update_bars(self, sample):
        # Assume channels 1 and 2 represent eye blinks
        eye_blink_signal = sample[0:2]
        eye_blink_count = sum(1 for value in eye_blink_signal if abs(value) > 1000)  # Threshold for blink detection

        # Fatigue level based on blink rate
        fatigue_level = eye_blink_count * 10  # Arbitrary scaling factor for visualization
        fatigue_height = min(280 * (fatigue_level / 1000) + 10, 290)  # Scale the value for the bar

        # Focus level based on alpha and beta power
        focus_level = self.calculate_focus_level(sample)
        focus_height = min(280 * (focus_level / 1000) + 10, 290)  # Scale the value for the bar

        self.fatigue_canvas.coords(self.fatigue_bar, 10, 290 - fatigue_height, 90, 290)
        self.focus_canvas.coords(self.focus_bar, 10, 290 - focus_height, 90, 290)

        if fatigue_level > 80:
            self.fatigue_canvas.itemconfig(self.fatigue_bar, fill="red")
        else:
            self.fatigue_canvas.itemconfig(self.fatigue_bar, fill="blue")

        if focus_level > 80:
            self.focus_canvas.itemconfig(self.focus_bar, fill="red")
        else:
            self.focus_canvas.itemconfig(self.focus_bar, fill="green")

    def calculate_focus_level(self, sample):
        # Use Welch's method to compute power spectral density
        fs = 250  # Sample rate
        freqs, psd = welch(sample, fs, nperseg=256)

        # Define frequency bands
        alpha_band = (8, 12)
        beta_band = (13, 30)

        # Calculate power in alpha and beta bands
        alpha_power = np.trapz(psd[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])])
        beta_power = np.trapz(psd[(freqs >= beta_band[0]) & (freqs <= beta_band[1])])

        focus_level = beta_power - alpha_power  # Arbitrary focus metric
        return focus_level

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGStreamApp(root)
    print("Starting EEG Stream App GUI...")
    root.mainloop()
