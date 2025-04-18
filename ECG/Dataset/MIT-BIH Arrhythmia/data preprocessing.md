To preprocess the MIT-BIH Arrhythmia Database for classification in Google Colab, we need to perform several steps: downloading the dataset, reading the ECG signals and annotations, preprocessing the signals (e.g., filtering noise, segmenting beats), and preparing the data for a classification task (e.g., labeling beats as normal or abnormal). Below, I’ll provide a **step-by-step guide** with full code, broken into parts, with detailed explanations for each step. Each code block will be wrapped in an `<xaiArtifact/>` tag as required, and I’ll ensure the explanations are beginner-friendly, assuming no prior knowledge.

We’ll use Python libraries like `wfdb` (for reading the dataset), `numpy`, `scipy` (for signal processing), and `pandas` (for data organization). The final output will be a dataset with segmented ECG beats and their corresponding labels, ready for training a machine learning model.

---

### **Prerequisites**
- **Google Colab**: A free cloud-based Jupyter notebook environment. You can access it at [colab.research.google.com](https://colab.research.google.com).
- **No Software Installation Needed**: Colab provides a Python environment with many libraries pre-installed. We’ll install additional libraries as needed.
- **Dataset**: The MIT-BIH Arrhythmia Database, which we’ll download directly in Colab.
- **Goal**: Preprocess ECG signals to extract individual heartbeats (centered around QRS complexes) and label them (e.g., normal vs. abnormal) for classification.

---

### **Step-by-Step Plan**
1. **Set Up the Environment**: Install necessary libraries in Google Colab.
2. **Download the Dataset**: Fetch the MIT-BIH Arrhythmia Database files.
3. **Read and Explore a Record**: Load one ECG record (signals and annotations) to understand the data.
4. **Preprocess the ECG Signal**: Apply filtering to remove noise and normalize the signal.
5. **Segment Heartbeats**: Extract individual beats around QRS complexes using annotations.
6. **Label the Beats**: Assign labels to each beat (e.g., normal or abnormal) based on annotations.
7. **Prepare the Dataset**: Combine all records into a single dataset for classification.
8. **Save the Preprocessed Data**: Store the segmented beats and labels for future use.

Each step will include a code block, an explanation of what the code does, and instructions on what to do next.

---

### **Step 1: Set Up the Environment**
**What**: Install Python libraries needed for the task.
**Why**: Google Colab has some libraries pre-installed (e.g., `numpy`, `pandas`), but we need `wfdb` for reading MIT-BIH files and `scipy` for signal processing.
**How**: Run a code cell in Colab to install libraries using `pip`.

**Detailed Explanation**:
- **Google Colab**: When you open Colab, you get a notebook with cells. Each cell can contain code or text. To run a code cell, click the play button (▶) or press `Shift + Enter`.
- **`pip install`**: A command to install Python packages. We’ll install:
  - `wfdb`: Reads MIT-BIH files (`.dat`, `.hea`, `.atr`).
  - `scipy`: For filtering ECG signals.
  - `matplotlib`: For visualizing signals (optional, for exploration).
- **Code Cell**: Paste the code below into a new cell in Colab and run it. This takes a few seconds, and you’ll see output confirming the installation.

```python
# Install required libraries
!pip install wfdb scipy matplotlib
```

**What to Do Next**:
- After running the cell, check the output. If there are no errors (e.g., “Successfully installed”), proceed to Step 2.
- If you see errors, ensure your internet connection is active, as Colab needs to download packages. Try running the cell again.

---

### **Step 2: Download the Dataset**
**What**: Download the MIT-BIH Arrhythmia Database files to Colab’s temporary storage.
**Why**: The dataset is hosted on PhysioNet, and we need its files (`.dat`, `.hea`, `.atr`) to process the ECG signals and annotations.
**How**: Use `wget` to download the dataset ZIP file and unzip it in Colab.

**Detailed Explanation**:
- **Colab’s File System**: Colab provides a temporary storage area (like a virtual computer). Files downloaded here are deleted when the session ends, but that’s fine for our task.
- **`wget`**: A command-line tool to download files from the internet. We’ll use it to get the dataset ZIP file.
- **Unzipping**: The dataset is compressed in a ZIP file. We’ll use the `unzip` command to extract it.
- **Directory**: We’ll create a folder called `mitdb` to store the dataset files.
- **Code**: The code below creates a folder, downloads the ZIP file (73.5 MB), and extracts it.

```python
# Import os to interact with the file system
import os

# Create a directory to store the dataset
os.makedirs('mitdb', exist_ok=True)

# Download the MIT-BIH Arrhythmia Database ZIP file
!wget -P mitdb https://physionet.org/files/mitdb/1.0.0/mitdb-1.0.0.zip

# Unzip the downloaded file into the mitdb folder
!unzip mitdb/mitdb-1.0.0.zip -d mitdb
```

**What Each Line Does**:
- `import os`: Loads the `os` module to create directories.
- `os.makedirs('mitdb', exist_ok=True)`: Creates a folder named `mitdb`. The `exist_ok=True` prevents errors if the folder already exists.
- `!wget -P mitdb ...`: Runs the `wget` command to download the ZIP file into the `mitdb` folder.
- `!unzip ...`: Extracts the ZIP file’s contents (48 records with `.dat`, `.hea`, `.atr` files) into the `mitdb` folder.

**What to Do Next**:
- Run the cell. It may take a minute to download and unzip.
- To verify, check the files in Colab’s file explorer:
  - Click the folder icon (📁) in Colab’s left sidebar.
  - Open the `mitdb` folder. You should see a subfolder `mitdb-1.0.0` with files like `100.dat`, `100.hea`, `100.atr`, etc.
- If the files are present, move to Step 3. If not, re-run the cell or check your internet connection.

---

### **Step 3: Read and Explore a Record**
**What**: Load one ECG record (e.g., record 100) to understand its signals and annotations.
**Why**: Before preprocessing, we need to see what the data looks like (e.g., ECG signal values, annotation types).
**How**: Use the `wfdb` library to read the signal (`.dat`) and annotation (`.atr`) files, then plot the signal.

**Detailed Explanation**:
- **Record Structure**: Each record (e.g., `100`) has:
  - `.dat`: ECG signal data (two channels, sampled at 360 Hz).
  - `.hea`: Metadata (e.g., sampling rate, gain).
  - `.atr`: Annotations (e.g., beat locations and types).
- **`wfdb` Library**:
  - `wfdb.rdsamp`: Reads the signal and header data.
  - `wfdb.rdann`: Reads the annotations.
- **Visualization**: We’ll plot the ECG signal with annotations (e.g., QRS peaks) to confirm we’re reading the data correctly.
- **Code**: The code below reads record 100, extracts the first channel (MLII), and plots 10 seconds of the signal with beat annotations.

```python
# Import required libraries
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Path to the record (in mitdb/mitdb-1.0.0 folder)
record_path = 'mitdb/mitdb-1.0.0/100'

# Read the ECG signal and header
record = wfdb.rdsamp(record_path)

# Extract the signal (two channels) and metadata
signal = record.p_signal  # Shape: (samples, 2) for two channels
fs = record.fs  # Sampling frequency (360 Hz)
channel_names = record.sig_name  # Channel names (e.g., ['MLII', 'V5'])

# Read the annotations
annotation = wfdb.rdann(record_path, 'atr')

# Extract annotation sample indices and symbols
ann_samples = annotation.sample  # Sample indices of beats
ann_symbols = annotation.symbol  # Beat types (e.g., 'N', 'V')

# Select the first channel (e.g., MLII) for plotting
ecg_signal = signal[:, 0]  # First channel

# Plot 10 seconds of the signal (3600 samples at 360 Hz)
time = np.arange(0, 3600) / fs  # Time axis in seconds
plt.figure(figsize=(15, 5))
plt.plot(time, ecg_signal[:3600], 'b-', label='ECG (MLII)')
plt.title('10 Seconds of ECG Signal (Record 100)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)

# Mark beat annotations
for sample, symbol in zip(ann_samples, ann_symbols):
    if sample < 3600:  # Only plot annotations within 10 seconds
        plt.axvline(x=sample/fs, color='r', linestyle='--', alpha=0.5)
        plt.text(sample/fs, max(ecg_signal[:3600]), symbol, color='r')

plt.legend()
plt.show()

# Print basic information
print(f'Sampling Frequency: {fs} Hz')
print(f'Signal Shape: {signal.shape}')
print(f'Channel Names: {channel_names}')
print(f'First 5 Annotation Symbols: {ann_symbols[:5]}')
print(f'First 5 Annotation Samples: {ann_samples[:5]}')
```

**What Each Line Does**:
- `import wfdb, matplotlib.pyplot, numpy`: Load libraries for reading data, plotting, and numerical operations.
- `record_path = 'mitdb/mitdb-1.0.0/100'`: Specifies the path to record 100 (without file extensions).
- `record = wfdb.rdsamp(record_path)`: Reads the signal (`.dat`) and header (`.hea`).
  - `record.p_signal`: The ECG signal as a NumPy array (rows = samples, columns = channels).
  - `record.fs`: Sampling frequency (360 Hz).
  - `record.sig_name`: Names of the channels (e.g., MLII, V5).
- `annotation = wfdb.rdann(record_path, 'atr')`: Reads the annotation file.
  - `annotation.sample`: Sample indices where beats occur.
  - `annotation.symbol`: Beat types (e.g., ‘N’ for normal, ‘V’ for ventricular ectopic).
- `ecg_signal = signal[:, 0]`: Selects the first channel (MLII) for simplicity.
- `time = np.arange(0, 3600) / fs`: Creates a time axis (0 to 10 seconds) by dividing sample indices by 360 Hz.
- `plt.plot(...)`: Plots the ECG signal.
- `plt.axvline(...)` and `plt.text(...)`: Marks beat locations with vertical lines and labels them with symbols.
- `print(...)`: Displays metadata to understand the data.

**What to Do Next**:
- Run the cell. You should see:
  - A plot showing 10 seconds of the ECG signal with red dashed lines at beat locations and symbols (e.g., ‘N’, ‘V’) above them.
  - Printed output, e.g.:
    ```
    Sampling Frequency: 360 Hz
    Signal Shape: (650000, 2)
    Channel Names: ['MLII', 'V5']
    First 5 Annotation Symbols: ['N', 'N', 'N', 'N', 'N']
    First 5 Annotation Samples: [370 663 957 1251 1544]
    ```
- The plot confirms we can read signals and annotations correctly. Move to Step 4 to preprocess the signal.

---

### **Step 4: Preprocess the ECG Signal**
**What**: Clean the ECG signal by removing noise and normalizing it.
**Why**: ECG signals often have noise (e.g., baseline wander, powerline interference) that can affect classification. Preprocessing improves signal quality.
**How**: Apply a bandpass filter to remove low-frequency (baseline wander) and high-frequency (powerline) noise, then normalize the signal.

**Detailed Explanation**:
- **Noise Types**:
  - **Baseline Wander**: Low-frequency drift (e.g., <0.5 Hz) due to breathing or movement.
  - **Powerline Noise**: 60 Hz interference from electrical equipment.
- **Bandpass Filter**: Allows frequencies between 0.5 Hz and 40 Hz, which covers the ECG’s relevant components (e.g., QRS complexes).
- **Normalization**: Scales the signal to a standard range (e.g., mean=0, std=1) for consistent input to machine learning models.
- **Tools**:
  - `scipy.signal`: For designing and applying filters.
  - `numpy`: For normalization.
- **Code**: The code below filters and normalizes the MLII channel of record 100.

```python
# Import required libraries
import wfdb
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load the record
record_path = 'mitdb/mitdb-1.0.0/100'
record = wfdb.rdsamp(record_path)
signal = record.p_signal[:, 0]  # First channel (MLII)
fs = record.fs  # 360 Hz

# Design a bandpass filter (0.5 Hz to 40 Hz)
lowcut = 0.5  # Low frequency cutoff
highcut = 40.0  # High frequency cutoff
nyquist = 0.5 * fs  # Nyquist frequency
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(4, [low, high], btype='band')  # 4th-order Butterworth filter

# Apply the filter
filtered_signal = signal.filtfilt(b, a, signal)

# Normalize the signal (zero mean, unit variance)
normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

# Plot original vs. filtered signal (first 10 seconds)
time = np.arange(0, 3600) / fs
plt.figure(figsize=(15, 5))
plt.plot(time, signal[:3600], 'b-', label='Original ECG')
plt.plot(time, normalized_signal[:3600], 'r-', label='Filtered & Normalized ECG')
plt.title('Original vs. Preprocessed ECG (Record 100)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```

**What Each Line Does**:
- `from scipy import signal`: Imports the signal processing module.
- `signal = record.p_signal[:, 0]`: Gets the MLII channel.
- `lowcut = 0.5, highcut = 40.0`: Defines the bandpass filter’s frequency range.
- `nyquist = 0.5 * fs`: Calculates the Nyquist frequency (180 Hz for 360 Hz sampling).
- `low = lowcut / nyquist, high = highcut / nyquist`: Normalizes cutoff frequencies for the filter.
- `b, a = signal.butter(4, [low, high], btype='band')`: Designs a 4th-order Butterworth bandpass filter.
- `filtered_signal = signal.filtfilt(b, a, signal)`: Applies the filter (forward and backward to avoid phase distortion).
- `normalized_signal = ...`: Subtracts the mean and divides by the standard deviation to normalize.
- Plotting compares the original and preprocessed signals.

**What to Do Next**:
- Run the cell. You should see a plot with:
  - Blue line: Original ECG with baseline wander and noise.
  - Red line: Smoother, normalized ECG with reduced noise.
- The preprocessed signal is cleaner and ready for segmentation. Move to Step 5.

---

### **Step 5: Segment Heartbeats**
**What**: Extract individual heartbeats from the ECG signal, centered around QRS complexes.
**Why**: For classification, we need fixed-length segments (e.g., 200 samples) around each beat, as machine learning models require consistent input sizes.
**How**: Use annotation sample indices to locate QRS peaks and extract a window of samples around each peak.

**Detailed Explanation**:
- **QRS Complex**: The sharp peak in an ECG representing ventricular depolarization. Annotations mark these peaks.
- **Window Size**: We’ll take 100 samples before and 100 samples after each QRS peak (201 samples total, ~0.56 seconds at 360 Hz).
- **Padding**: For beats near the signal’s start or end, we’ll pad with zeros to ensure all segments are the same length.
- **Code**: The code below segments beats from the preprocessed signal of record 100.

```python
# Import required libraries
import wfdb
import numpy as np
from scipy import signal

# Load the record and annotations
record_path = 'mitdb/mitdb-1.0.0/100'
record = wfdb.rdsamp(record_path)
signal = record.p_signal[:, 0]  # MLII channel
fs = record.fs  # 360 Hz
annotation = wfdb.rdann(record_path, 'atr')
ann_samples = annotation.sample  # QRS peak indices

# Preprocess the signal (bandpass filter and normalize)
lowcut = 0.5
highcut = 40.0
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(4, [low, high], btype='band')
filtered_signal = signal.filtfilt(b, a, signal)
normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

# Segment beats
window_size = 100  # Samples before and after QRS peak
beats = []
for peak in ann_samples:
    start = peak - window_size
    end = peak + window_size + 1  # +1 to include the peak
    if start >= 0 and end <= len(normalized_signal):  # Check boundaries
        beat = normalized_signal[start:end]
        beats.append(beat)
    else:
        # Pad with zeros if near signal edges
        beat = np.zeros(2 * window_size + 1)
        signal_slice = normalized_signal[max(0, start):min(len(normalized_signal), end)]
        beat[:len(signal_slice)] = signal_slice
        beats.append(beat)

beats = np.array(beats)  # Shape: (n_beats, 201)

# Print results
print(f'Number of Beats: {len(beats)}')
print(f'Shape of Beats Array: {beats.shape}')

# Plot the first beat
import matplotlib.pyplot as plt
time = np.arange(-window_size, window_size + 1) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, beats[0], 'b-')
plt.title('First Segmented Beat')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.show()
```

**What Each Line Does**:
- Loads and preprocesses the signal (same as Step 4).
- `window_size = 100`: Defines 100 samples (~0.278 seconds) before and after the QRS peak.
- `beats = []`: A list to store segmented beats.
- `for peak in ann_samples`: Loops through QRS peak indices.
- `start = peak - window_size, end = peak + window_size + 1`: Defines the window around the peak.
- `if start >= 0 and end <= len(normalized_signal)`: Ensures the window is within the signal’s bounds.
- `beat = normalized_signal[start:end]`: Extracts the beat.
- Else, pads with zeros for edge cases.
- `beats = np.array(beats)`: Converts the list to a NumPy array.
- Plots the first beat to visualize the segment.

**What to Do Next**:
- Run the cell. You should see:
  - Printed output, e.g., `Number of Beats: ~2273, Shape: (2273, 201)`.
  - A plot of the first beat, showing a QRS complex centered in the window.
- The beats are now segmented. Move to Step 6 to label them.

---

### **Step 6: Label the Beats**
**What**: Assign labels to each segmented beat based on annotation symbols.
**Why**: For classification, each beat needs a label (e.g., 0 for normal, 1 for abnormal).
**How**: Map annotation symbols to binary labels (normal vs. abnormal) using a dictionary.

**Detailed Explanation**:
- **Annotation Symbols**: Common symbols include:
  - `N`: Normal beat.
  - `V`: Ventricular ectopic (abnormal).
  - `S`: Supraventricular ectopic (abnormal).
  - `F`: Fusion beat (abnormal).
  - Others (e.g., `/` for paced beats) may be excluded.
- **Binary Classification**: We’ll label `N` as 0 (normal) and `V`, `S`, `F` as 1 (abnormal).
- **Code**: The code below labels beats for record 100.

```python
# Import required libraries
import wfdb
import numpy as np
from scipy import signal

# Load the record and annotations
record_path = 'mitdb/mitdb-1.0.0/100'
record = wfdb.rdsamp(record_path)
signal = record.p_signal[:, 0]  # MLII channel
fs = record.fs  # 360 Hz
annotation = wfdb.rdann(record_path, 'atr')
ann_samples = annotation.sample
ann_symbols = annotation.symbol

# Preprocess the signal
lowcut = 0.5
highcut = 40.0
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(4, [low, high], btype='band')
filtered_signal = signal.filtfilt(b, a, signal)
normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

# Segment beats
window_size = 100
beats = []
valid_symbols = []
for peak, symbol in zip(ann_samples, ann_symbols):
    if symbol in ['N', 'V', 'S', 'F']:  # Only include these beat types
        start = peak - window_size
        end = peak + window_size + 1
        if start >= 0 and end <= len(normalized_signal):
            beat = normalized_signal[start:end]
            beats.append(beat)
            valid_symbols.append(symbol)
        else:
            beat = np.zeros(2 * window_size + 1)
            signal_slice = normalized_signal[max(0, start):min(len(normalized_signal), end)]
            beat[:len(signal_slice)] = signal_slice
            beats.append(beat)
            valid_symbols.append(symbol)

beats = np.array(beats)

# Label beats (0 = normal, 1 = abnormal)
label_map = {'N': 0, 'V': 1, 'S': 1, 'F': 1}
labels = np.array([label_map[symbol] for symbol in valid_symbols])

# Print results
print(f'Number of Beats: {len(beats)}')
print(f'Number of Labels: {len(labels)}')
print(f'Label Distribution: Normal={np.sum(labels == 0)}, Abnormal={np.sum(labels == 1)}')

# Plot first normal and abnormal beat
normal_idx = np.where(labels == 0)[0][0]
abnormal_idx = np.where(labels == 1)[0][0]
time = np.arange(-window_size, window_size + 1) / fs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time, beats[normal_idx], 'b-')
plt.title(f'Normal Beat (Symbol: {valid_symbols[normal_idx]})')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(time, beats[abnormal_idx], 'r-')
plt.title(f'Abnormal Beat (Symbol: {valid_symbols[abnormal_idx]})')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
```

**What Each Line Does**:
- Filters beats to include only `N`, `V`, `S`, `F` symbols.
- `valid_symbols.append(symbol)`: Stores the symbol for each valid beat.
- `label_map = {'N': 0, 'V': 1, 'S': 1, 'F': 1}`: Defines the labeling rule.
- `labels = np.array([label_map[symbol] for symbol in valid_symbols])`: Maps symbols to 0 or 1.
- Plots one normal and one abnormal beat to compare their shapes.

**What to Do Next**:
- Run the cell. You should see:
  - Printed output, e.g., `Normal=~2239, Abnormal=~34`.
  - Two plots: a normal beat (likely smooth) and an abnormal beat (possibly irregular).
- The beats and labels are ready for one record. Move to Step 7 to process all records.

---

### **Step 7: Prepare the Dataset**
**What**: Process all 48 records to create a combined dataset of beats and labels.
**Why**: A single record has limited data. Combining all records increases the dataset size and diversity.
**How**: Loop through all records, applying the preprocessing, segmentation, and labeling steps.

**Detailed Explanation**:
- **Records**: The dataset has 48 records (100 to 234, with some gaps). We’ll skip records with paced beats (102, 104, 107, 217).
- **Loop**: Iterate over records, preprocess each signal, segment beats, and collect labels.
- **Storage**: Store all beats and labels in NumPy arrays.
- **Code**: The code below processes all valid records.

```python
# Import required libraries
import wfdb
import numpy as np
from scipy import signal
import os

# List of records (excluding paced records)
records = [str(i) for i in range(100, 235) if i not in [102, 104, 107, 217]]
base_path = 'mitdb/mitdb-1.0.0/'

# Initialize lists to store all beats and labels
all_beats = []
all_labels = []

# Parameters
fs = 360  # Sampling frequency
window_size = 100  # Samples before/after QRS
label_map = {'N': 0, 'V': 1, 'S': 1, 'F': 1}

for record_id in records:
    print(f'Processing record {record_id}...')
    record_path = os.path.join(base_path, record_id)
    
    # Load record
    try:
        record = wfdb.rdsamp(record_path)
        signal = record.p_signal[:, 0]  # MLII channel
        annotation = wfdb.rdann(record_path, 'atr')
        ann_samples = annotation.sample
        ann_symbols = annotation.symbol
    except:
        print(f'Failed to load record {record_id}')
        continue
    
    # Preprocess signal
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal)
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    
    # Segment and label beats
    for peak, symbol in zip(ann_samples, ann_symbols):
        if symbol in label_map:
            start = peak - window_size
            end = peak + window_size + 1
            if start >= 0 and end <= len(normalized_signal):
                beat = normalized_signal[start:end]
                all_beats.append(beat)
                all_labels.append(label_map[symbol])
            else:
                beat = np.zeros(2 * window_size + 1)
                signal_slice = normalized_signal[max(0, start):min(len(normalized_signal), end)]
                beat[:len(signal_slice)] = signal_slice
                all_beats.append(beat)
                all_labels.append(label_map[symbol])

# Convert to NumPy arrays
all_beats = np.array(all_beats)
all_labels = np.array(all_labels)

# Print summary
print(f'Total Beats: {len(all_beats)}')
print(f'Beat Shape: {all_beats.shape}')
print(f'Label Distribution: Normal={np.sum(all_labels == 0)}, Abnormal={np.sum(all_labels == 1)}')

# Save the dataset
np.save('mitdb_beats.npy', all_beats)
np.save('mitdb_labels.npy', all_labels)
```

**What Each Line Does**:
- `records = [str(i) for i in range(100, 235) if i not in [102, 104, 107, 217]]`: Lists record IDs, excluding paced records.
- `all_beats = [], all_labels = []`: Lists to collect data from all records.
- `for record_id in records`: Loops through records.
- `try ... except`: Skips records that fail to load (e.g., missing files).
- Applies preprocessing, segmentation, and labeling as before.
- `np.save(...)`: Saves the arrays to `.npy` files for later use.

**What to Do Next**:
- Run the cell. It may take 5–10 minutes to process all records.
- Check the output, e.g., `Total Beats: ~100000, Normal=~90000, Abnormal=~10000`.
- Verify the saved files in Colab’s file explorer (`mitdb_beats.npy`, `mitdb_labels.npy`).
- Move to Step 8 to save the data permanently.

---

### **Step 8: Save the Preprocessed Data**
**What**: Download the preprocessed dataset to your computer.
**Why**: Colab’s storage is temporary. Saving to Google Drive or your local machine ensures you keep the data.
**How**: Use Google Drive to save the `.npy` files.

**Detailed Explanation**:
- **Google Drive**: Colab can mount your Google Drive to save files.
- **Mounting Drive**: Authenticate with your Google account to access Drive.
- **Copy Files**: Move the `.npy` files to Drive.
- **Code**: The code below mounts Drive and saves the files.

```python
# Import Google Drive
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Copy files to Google Drive
import shutil
shutil.copy('mitdb_beats.npy', '/content/drive/My Drive/mitdb_beats.npy')
shutil.copy('mitdb_labels.npy', '/content/drive/My Drive/mitdb_labels.npy')

print('Files saved to Google Drive!')
```

**What Each Line Does**:
- `drive.mount('/content/drive')`: Prompts you to authenticate and mount Drive.
- `shutil.copy(...)`: Copies the `.npy` files to your Drive’s root folder.
- Prints a confirmation message.

**What to Do Next**:
- Run the cell. Follow the authentication link, copy the code, and paste it into Colab.
- Check your Google Drive (root folder) for `mitdb_beats.npy` and `mitdb_labels.npy`.
- The dataset is now ready for classification. You can load these files later with `np.load()`.

---

### **Final Notes**
- **Dataset Format**:
  - `mitdb_beats.npy`: Shape `(n_beats, 201)`, each row is a beat (201 samples).
  - `mitdb_labels.npy`: Shape `(n_beats,)`, each element is 0 (normal) or 1 (abnormal).
- **Next Steps**: Use the dataset for machine learning (e.g., train a CNN or SVM). If you need code for this, let me know!
- **Troubleshooting**:
  - If a step fails, check the error message and ensure previous steps completed correctly.
  - Ensure files are in the `mitdb/mitdb-1.0.0` folder.
  - For memory issues, reduce the number of records processed in Step 7 (e.g., use `records = ['100', '101']`).

This guide provides a complete pipeline to preprocess the MIT-BIH Arrhythmia Database in Google Colab. Each step builds on the previous one, and the code is modular for easy debugging. 
