from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet
import numpy as np

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)  # uV/count
SCALE_FACTOR_AUX = 0.002 / (2**4)

print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")
info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 16, 250, 'float32', 'OpenBCItestEEG')

outlet_eeg = StreamOutlet(info_eeg)

def lsl_streamers(sample):
    outlet_eeg.push_sample(np.array(sample.channels_data) * SCALE_FACTOR_EEG)

print("Initializing OpenBCI Cyton Board with Daisy")
board = OpenBCICyton(port='COM6', daisy=True) 
print("Starting EEG data stream...")
board.start_stream(lsl_streamers)
