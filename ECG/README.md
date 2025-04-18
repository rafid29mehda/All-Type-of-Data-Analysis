

# 📄 All About ECG Data: Format, Analysis, Processing, and Machine Learning Integration

---

## ✅ 1. What is ECG Data?

**ECG (Electrocardiogram)** is a medical recording of the electrical activity of the heart over a period of time. It’s used to monitor heart conditions like arrhythmias, heart attacks, and more.

**Key components of an ECG waveform:**
- **P wave** – Atrial depolarization
- **QRS complex** – Ventricular depolarization
- **T wave** – Ventricular repolarization

---

## ✅ 2. ECG Data Formats

ECG data can come in various formats depending on the device and dataset. Some popular formats are:

### 🧾 Common ECG Formats:

| Format | Description |
|--------|-------------|
| **.mat** | MATLAB file format (common in PhysioNet datasets) |
| **.csv** | Simple comma-separated format, each row = data point |
| **.edf** | European Data Format, often used in biosignals |
| **.xml / .txt** | Raw signals with timestamp or metadata |
| **.hea** | Header files with sampling frequency, labels, etc. (used with .dat in PhysioNet) |

### 📦 Example: MIT-BIH Dataset (PhysioNet)
- `.dat` – Raw binary ECG data
- `.hea` – Metadata: sample rate, duration, lead info
- `.atr` – Annotations (like beat type, rhythm)

---

## ✅ 3. How to Read ECG Data

### 🛠️ Tools & Libraries:
- **Python**
  - `wfdb` – Read PhysioNet ECG datasets
  - `scipy.io` – For .mat files
  - `pandas` – For CSV files
  - `biosppy` – For signal processing
  - `neurokit2` – For analyzing physiological signals

### 🔍 Example: Reading from PhysioNet (.dat + .hea)

```python
import wfdb

# Load ECG record
record = wfdb.rdrecord('100', sampfrom=0, sampto=1000, pn_dir='mitdb')
wfdb.plot_wfdb(record=record, title='ECG signal')
```

---

# ✅ 1. ECG Data Formats (In-Depth)

ECG data is collected by medical devices that record the **electrical signals of the heart**. Depending on the manufacturer, clinical setup, or public dataset, ECG data is saved in different file formats.

### 🔍 Key Goals of ECG Data Formats:
- Store raw ECG waveform signals
- Include **metadata** like sampling rate, lead names
- Sometimes include **annotations** (e.g., arrhythmia labels, beat types)

---

## 📦 Common ECG File Formats

### 1. `.dat` + `.hea` (Used in PhysioNet/MIT-BIH)
- **.dat**: Binary file containing raw ECG signal
- **.hea**: Text-based header file
  - Sampling frequency
  - Number of signals (channels/leads)
  - Units (mV, etc.)
  - Signal lengths
  - Patient ID
- **.atr** or **.qrs**: Annotation files with labels (e.g., R-peaks, rhythm)

✅ **Example (100.hea file):**
```
100 2 360 650000
100.dat 212 200 11 1024 928 -474 0 MLII
100.dat 212 200 11 1024 928 -474 0 V5
# 2 leads, 360 Hz sampling rate, 650000 samples
```

---

### 2. `.mat` (MATLAB format)
- Stores signals as matrices (common in deep learning datasets)
- Contains:
  - ECG waveform as an array
  - Sampling frequency
  - Optional annotations or labels
- Can be read in **Python** using `scipy.io`

✅ Example:
```python
from scipy.io import loadmat
mat = loadmat('ecg_record.mat')
print(mat.keys())
```

---

### 3. `.csv` (Comma-Separated Values)
- Easy to read/edit manually or in Python
- Rows represent time points, columns represent leads/signals
- May include timestamp or sample index

✅ Example CSV structure:
```
Time, Lead1, Lead2
0.00, 0.15, 0.22
0.01, 0.13, 0.21
```

---

### 4. `.edf` (European Data Format)
- Common in sleep studies and biomedical signals
- Stores multiple signal channels
- Includes patient info, timestamps, labels
- Readable with `pyEDFlib` in Python

---

### 5. `.xml`, `.json`, `.txt`
- Sometimes used by hospital-grade ECG machines
- XML/JSON includes metadata + signal + annotations
- `.txt` often used for raw values without structure

---

## 📚 Summary Table

| Format | Description | Libraries to Read | Used in |
|--------|-------------|-------------------|---------|
| `.dat` + `.hea` | Binary signal + header | `wfdb` | PhysioNet |
| `.mat` | MATLAB matrix format | `scipy.io` | China ECG, PhysioNet |
| `.csv` | Easy table format | `pandas`, `numpy` | Custom datasets |
| `.edf` | Multi-signal format | `pyEDFlib` | Sleep/EEG/ECG |
| `.xml`, `.json` | Structured metadata | `xml.etree`, `json` | Vendor-specific |

---

# ✅ 2. How to Read ECG Data (Step-by-Step)


---

### 🔹 A. Reading PhysioNet (.dat + .hea + .atr) with `wfdb`

Install the library:

```bash
pip install wfdb
```

#### ✅ Load an ECG record:
```python
import wfdb

# Load ECG signal (360 Hz, 2 channels)
record = wfdb.rdrecord('100', pn_dir='mitdb')
print(record.__dict__)
```

#### ✅ Plot ECG signal:
```python
wfdb.plot_wfdb(record=record, title='ECG Signal - MITDB Record 100')
```

#### ✅ Load annotations (e.g., R-peaks, beat labels):
```python
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
print(annotation.symbol)  # e.g., ['N', 'V', 'A']
```

---

### 🔹 B. Reading `.mat` (MATLAB) ECG Files

```python
from scipy.io import loadmat

mat = loadmat('ecg_data.mat')
ecg_signal = mat['val']  # or check keys with mat.keys()
```

Tip: Normalize the signal for ML:

```python
normalized = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
```

---

### 🔹 C. Reading `.csv` ECG Files

```python
import pandas as pd

df = pd.read_csv('ecg_data.csv')
print(df.head())

# Plot lead
import matplotlib.pyplot as plt
plt.plot(df['Lead1'])
plt.title("ECG - Lead1")
plt.show()
```

---

### 🔹 D. Reading `.edf` ECG Files

```python
import pyedflib

f = pyedflib.EdfReader('ecg_record.edf')
n = f.signals_in_file
signal_labels = f.getSignalLabels()

for i in range(n):
    ecg = f.readSignal(i)
```

---

## ⚠️ Important Notes:
- Always check **sampling rate** (`fs` or `frequency`) from metadata
- Some records are multi-lead (e.g., Lead I, II, III) — treat each column
- Use **filters** before ML modeling (remove noise, baseline wander)

---

## ✅ Example Workflow (MIT-BIH):

```python
import wfdb

# Load ECG + annotations
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# Plot with annotations
wfdb.plot_wfdb(record=record, annotation=annotation,
               title='MIT-BIH ECG with Annotations')
```


---

## ✅ 4. How to Analyze ECG Data

### 🧠 Main Goals of ECG Analysis:
- Detect abnormalities (e.g., arrhythmia)
- Extract heart rate, RR interval
- Identify P, QRS, and T wave components

### 🧰 Techniques:
- **Peak detection** (R-peaks using Pan-Tompkins algorithm)
- **Feature extraction** (RR intervals, wave durations)
- **Beat classification** (normal vs abnormal)
- **Signal quality check** (noise, missing data)

### Example:
```python
import neurokit2 as nk

# Simulated ECG
ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=1000)
# Process
signals, info = nk.ecg_process(ecg_signal, sampling_rate=1000)
# Visualize
nk.ecg_plot(signals, info)
```

---

## ✅ 5. How to Preprocess ECG Data

### ⚙️ Preprocessing Steps:

| Step | Purpose |
|------|---------|
| **Resampling** | Normalize sampling rate |
| **Filtering** | Remove noise (baseline wander, powerline) |
| **Normalization** | Scale data between 0–1 or standardize |
| **Segmentation** | Divide into windows/heartbeats |
| **Feature extraction** | Use PQRST, RR intervals, etc. |

### Filters:
- **Low-pass**: remove high-frequency noise
- **High-pass**: remove baseline drift
- **Bandpass**: keep useful heart frequencies (e.g., 0.5–40Hz)

### Example:
```python
from scipy.signal import butter, filtfilt

# Bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

filtered_ecg = butter_bandpass_filter(ecg_signal, 0.5, 40, 1000)
```

---

## ✅ 6. Integrating ECG Data in Machine Learning

### 💡 Machine Learning Use Cases:
- **Arrhythmia detection**
- **Sleep stage classification**
- **Stress/emotion detection**
- **Biometric identification**

### 🧾 Input Options for ML Models:
1. **Raw signal** (as time-series input)
2. **Engineered features** (e.g., RR interval, HRV)
3. **Images** (Convert to spectrograms or beat plots)

### 📊 Common ML Models:
| Model | Use Case |
|-------|----------|
| **Random Forest, SVM** | With hand-crafted features |
| **CNN** | Image-based or raw 1D signals |
| **RNN / LSTM** | Time-series classification |
| **Autoencoders** | Anomaly detection |
| **Transformers** | Long ECG sequences |

### Example: Simple Classifier Using Features

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = features_df.drop('label', axis=1)
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

---

## ✅ 7. Useful Datasets for ECG ML Projects

| Dataset | Source | Description |
|---------|--------|-------------|
| **MIT-BIH Arrhythmia** | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) | Classic for heartbeat classification |
| **PTB Diagnostic ECG** | [PhysioNet](https://physionet.org/content/ptbdb/1.0.0/) | Multiple diseases |
| **INCART 12-lead ECG** | [PhysioNet](https://physionet.org/content/incartdb/1.0.0/) | 75 annotated records |
| **Chapman-Shaoxing** | [PhysioNet](https://physionet.org/content/chapman-shaoxing/1.0.0/) | Large 12-lead ECG |

---

## ✅ 8. Tools for Visualization & GUI

- **ECG-kit (MATLAB)**
- **PhysioNet LightWave viewer**
- **Python: Matplotlib, WFDB, Neurokit2**

---

## ✅ 9. Summary

| Step | Description |
|------|-------------|
| **Understand Format** | .csv, .mat, .hea/.dat |
| **Load & Plot** | Use wfdb, pandas, matplotlib |
| **Analyze** | R-peak detection, waveform segmentation |
| **Preprocess** | Filter, normalize, extract features |
| **Model** | Use ML (Random Forest, CNN, LSTM) |
| **Evaluate** | Accuracy, F1-score, AUC, etc. |

---

## 🧠 Real-Life Example

**Example:** Arrhythmia Detection  
You can take the MIT-BIH dataset, extract R-R intervals and QRS durations, then train a Random Forest model to classify beats into types like normal, premature ventricular contraction, etc.

---


## 🔗 Reference Links:
- PhysioNet Datasets: https://physionet.org/about/database/
- wfdb Python Docs: https://wfdb.readthedocs.io/en/latest/
- NeuroKit2: https://neuropsychology.github.io/NeuroKit/

