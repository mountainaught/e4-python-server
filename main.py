import asyncio
import struct
import time
import math
import argparse
import os
import csv
from datetime import datetime
from bleak import BleakClient, BleakScanner

# --- Empatica E4 Constants ---
DEVICE_NAME = "Empatica E4"
BVP_CHARACTERISTIC_UUID = "00003ea1-0000-1000-8000-00805f9b34fb"
GSR_CHARACTERISTIC_UUID = "00003ea8-0000-1000-8000-00805f9b34fb"
ACC_CHARACTERISTIC_UUID = "00003ea3-0000-1000-8000-00805f9b34fb"
ST_CHARACTERISTIC_UUID = "00003ea6-0000-1000-8000-00805f9b34fb"
BATTERY_CHARACTERISTIC_UUID = "00003eb3-0000-1000-8000-00805f9b34fb"
CMD_CHARACTERISTIC_UUID = "00003e71-0000-1000-8000-00805f9b34fb"

# --- Data Manager for CSV Saving ---

class DataManager:
    """Handles creating, writing to, and closing CSV files for a session."""
    def __init__(self):
        self.session_path = ""
        self.writers = {}
        self.files = {}
        self.create_session_folder()

    def create_session_folder(self):
        if not os.path.exists('output'):
            os.makedirs('output')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_path = os.path.join('output', timestamp)
        os.makedirs(self.session_path)
        print(f"Saving data to: {self.session_path}")

    def setup_file(self, filename, initial_timestamp, sample_rate):
        filepath = os.path.join(self.session_path, filename)
        self.files[filename] = open(filepath, 'w', newline='')
        self.writers[filename] = csv.writer(self.files[filename])
        if "ACC" in filename:
             self.writers[filename].writerow([f"{initial_timestamp:.6f}", f"{initial_timestamp:.6f}", f"{initial_timestamp:.6f}"])
             self.writers[filename].writerow([f"{sample_rate:.6f}", f"{sample_rate:.6f}", f"{sample_rate:.6f}"])
        else:
            self.writers[filename].writerow([f"{initial_timestamp:.6f}"])
            self.writers[filename].writerow([f"{sample_rate:.6f}"])

    def write_data(self, filename, data_row):
        if filename in self.writers:
            self.writers[filename].writerow(data_row)

    def close_files(self):
        for f in self.files.values():
            f.close()
        print("CSV files saved and closed.")

# --- Post-session Graphing ---

def generate_graphs_from_csv(session_path):
    """Reads data from CSV files, generates plots, and saves them to a file."""
    print("Generating graphs from saved data...")
    try:
        import pandas as pd
        import matplotlib
        # Use a non-interactive backend to be safe
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nCould not import pandas or matplotlib. Please install with: pip install pandas matplotlib")
        return

    files = {'BVP.csv': 'BVP', 'EDA.csv': 'EDA (µS)', 'TEMP.csv': 'Temperature (°C)', 'ACC.csv': 'Accelerometer (g)'}
    fig, axs = plt.subplots(len(files), 1, figsize=(12, 16), constrained_layout=True)
    fig.suptitle(f'Empatica E4 Data Session: {os.path.basename(session_path)}', fontsize=16)

    for i, (filename, title) in enumerate(files.items()):
        filepath = os.path.join(session_path, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50:
            try:
                df = pd.read_csv(filepath, skiprows=2, header=None)
                if filename == 'ACC.csv':
                    axs[i].plot(df[0] / 64.0, label='X'); axs[i].plot(df[1] / 64.0, label='Y'); axs[i].plot(df[2] / 64.0, label='Z')
                    axs[i].legend()
                else:
                    axs[i].plot(df[0])
                axs[i].set_title(title); axs[i].grid(True)
            except Exception as e:
                axs[i].set_title(f"Could not plot {title}: {e}")
        else:
            axs[i].set_title(f"{title} (No data collected)")
    
    graph_output_path = os.path.join(session_path, 'summary_graphs.png')
    plt.savefig(graph_output_path)
    print(f"Graphs saved to: {graph_output_path}")


# --- Data Handling & BLE ---

class StreamProcessor:
    def __init__(self, name, filename, sampling_rate, data_manager=None, print_mode=False, multiplier=1.0):
        self.name = name; self.filename = filename; self.sampling_rate = float(sampling_rate)
        self.data_manager = data_manager; self.print_mode = print_mode; self.multiplier = multiplier
        self.first_packet = True

    def process_packet(self, data):
        offset = 0
        if self.first_packet and self.data_manager:
            try:
                ts = struct.unpack_from('<f', data, 0)[0]
                self.data_manager.setup_file(self.filename, ts, self.sampling_rate)
                self.first_packet = False; offset = 4
            except struct.error: return

        i = offset
        while i + 4 <= len(data):
            try:
                value = struct.unpack_from('<f', data, i)[0]
                if not math.isnan(value) and abs(value) < 1e6:
                    if self.data_manager: self.data_manager.write_data(self.filename, [value * self.multiplier])
                    if self.print_mode: print(f"{self.name}: {value * self.multiplier:.4f}")
                i += 4
            except struct.error: break

def handle_acc(sender, data, data_manager=None, print_mode=False):
    i = 0
    while i + 3 <= len(data):
        try:
            x, y, z = struct.unpack_from('<bbb', data, i)
            if data_manager: data_manager.write_data('ACC.csv', [x, y, z])
            if print_mode: print(f"Accelerometer: X: {x/64.0:.2f}g, Y: {y/64.0:.2f}g, Z: {z/64.0:.2f}g")
            i += 3
        except struct.error: break

def handle_battery(sender, data, print_mode=False):
    if not print_mode: return
    try:
        level = struct.unpack('<B', data)[0]
        print(f"Battery: {level}%")
    except struct.error: pass

async def main_ble_loop(args, data_manager):
    """The main asynchronous part of the program."""
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    if not device:
        print(f"Could not find a device named {DEVICE_NAME}"); return
    print(f"Found {DEVICE_NAME} with address {device.address}. Connecting...")
    
    bvp_processor = StreamProcessor("BVP", "BVP.csv", 64, data_manager, args.print)
    gsr_processor = StreamProcessor("GSR", "EDA.csv", 4, data_manager, args.print, multiplier=1000)
    temp_processor = StreamProcessor("Temp", "TEMP.csv", 4, data_manager, args.print)
    acc_first_packet = True

    async def acc_handler(sender, data):
        nonlocal acc_first_packet
        if acc_first_packet and data_manager:
            data_manager.setup_file('ACC.csv', time.time(), 32.0)
            acc_first_packet = False
        handle_acc(sender, data, data_manager, args.print)

    async with BleakClient(device) as client:
        if not client.is_connected:
            print("Failed to connect."); return
        print(f"Connected to {DEVICE_NAME}\nInitializing...")
        
        await client.start_notify(BVP_CHARACTERISTIC_UUID, lambda s, d: bvp_processor.process_packet(d))
        await client.start_notify(GSR_CHARACTERISTIC_UUID, lambda s, d: gsr_processor.process_packet(d))
        await client.start_notify(ST_CHARACTERISTIC_UUID, lambda s, d: temp_processor.process_packet(d))
        await client.start_notify(ACC_CHARACTERISTIC_UUID, acc_handler)
        await client.start_notify(BATTERY_CHARACTERISTIC_UUID, lambda s, d: handle_battery(s, d, args.print))
        
        command = struct.pack('<BI', 1, int(time.time()))
        await client.write_gatt_char(CMD_CHARACTERISTIC_UUID, command)

        print("\n--- Data Streaming Started ---")
        print("Press Ctrl+C to disconnect and stop.")

        while True: await asyncio.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect, stream, and save data from an Empatica E4.")
    parser.add_argument('--save', action='store_true', help="Save sensor data to CSV files in a timestamped folder.")
    parser.add_argument('--print', action='store_true', help="Print sensor data to the console in real-time.")
    parser.add_argument('--graph', action='store_true', help="Generate and save graphs from data after the session ends. Requires --save.")
    args = parser.parse_args()

    if args.graph and not args.save:
        parser.error("--graph requires --save to be enabled.")

    data_manager = DataManager() if args.save else None
    
    try:
        asyncio.run(main_ble_loop(args, data_manager))
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        if data_manager:
            data_manager.close_files()
            if args.graph:
                generate_graphs_from_csv(data_manager.session_path)