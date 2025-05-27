#### Introduction to WFDB
WFDB, or WaveForm DataBase, is a software package and file format standard developed at MIT for handling biomedical signals, such as ECG, EEG, and other physiological data. It has been widely used for decades in biomedical research, clinical studies, and education, making it an essential tool for the goals. The WFDB format includes:
- **Header files (.hea)**: Contain metadata like sampling frequency, number of channels, and signal names.
- **Data files (.dat)**: Store the raw signal data.
- **Annotation files (.atr, .qrs, etc.)**: Include labels for events, such as beat annotations in ECG data, which are critical for supervised learning tasks.

The WFDB Python package, which we'll focus on, provides a library of tools for reading, writing, and processing these signals and annotations, aligning with the need to work with ECG data in research.

#### Installation and Setup
To get started, install the WFDB Python package using pip:
```bash
pip install wfdb
```
This package is hosted on PyPI and includes demo scripts and example data, accessible via its [GitHub repository](https://github.com/MIT-LCP/wfdb-python). Ensure we have Python installed, and consider using Google Colab for free cloud-based computing, especially for beginners, as it supports all necessary libraries like `numpy`, `pandas`, and `matplotlib` for visualization.

#### Key Functions and Operations
WFDB offers several core functions for handling biomedical signals. Below is a table summarizing the key functions, their purposes, and examples, tailored for ECG data processing:

| **Function/Class**          | **Purpose**                                      | **Example**                                                                 |
|-----------------------------|--------------------------------------------------|-----------------------------------------------------------------------------|
| `wfdb.rdheader`             | Read header file (metadata)                     | `header = wfdb.rdheader('100', pn_dir='path/to/mitdb')`                     |
| `wfdb.rdrecord`             | Read record (signal + header)                   | `record = wfdb.rdrecord('100', pn_dir='path/to/mitdb')`                     |
| `wfdb.rdsamp`               | Read physical signals and key fields            | `signals, fields = wfdb.rdsamp('100', pn_dir='path/to/mitdb')`              |
| `wfdb.rdann`                | Read annotation file (e.g., beat labels)        | `ann = wfdb.rdann('100', 'atr', pn_dir='path/to/mitdb')`                    |
| `wfdb.wrsamp`               | Write record (create header and data files)     | `wfdb.wrsamp('my_ecg', fs=360, units=['mV', 'mV'], p_signal=signals)`       |
| `wfdb.wrann`                | Write annotation file                           | `wfdb.wrann('my_ecg', 'atr', samples, symbols)`                             |
| `wfdb.Record`               | Create and initialize a single-segment record   | `record = wfdb.Record(record_name='r1', fs=250, n_sig=2, sig_len=1000)`     |
| `wfdb.MultiRecord`          | Handle multi-segment records                    | `record_m = wfdb.MultiRecord(record_name='rm', fs=50, n_sig=8, sig_len=9999)`|

These functions allow we to read existing ECG data, like from the MIT-BIH Arrhythmia Database, and write the own data for analysis, which is essential for research projects.

#### Working with ECG Data: Practical Examples
Given the focus on ECG-related work, let's dive into practical examples using the MIT-BIH Arrhythmia Database, a standard dataset for ECG research, available at [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). This database contains 48 half-hour ECG recordings with annotations for arrhythmias, making it ideal for practice.

##### Example 1: Reading and Plotting an ECG Record
To read and visualize an ECG signal:
```python
import wfdb
import matplotlib.pyplot as plt

# Read record '100' from MIT-BIH
record = wfdb.rdrecord('100', pn_dir='path/to/mitdb')

# Plot the first channel (first 1000 samples)
plt.figure(figsize=(10, 4))
plt.plot(record.p_signal[:1000, 0])
plt.title('ECG Signal - Record 100')
plt.xlabel('Samples')
plt.ylabel('Amplitude (mV)')
plt.show()
```
This code reads the record, accesses the physical signal (`record.p_signal`), and plots it, helping we visualize the ECG waveform.

##### Example 2: Reading and Using Annotations
Annotations are crucial for labeling beats. To read and explore annotations:
```python
ann = wfdb.rdann('100', 'atr', pn_dir='path/to/mitdb')
print(ann.sample)  # Sample indices of annotations
print(ann.symbol)  # Annotation symbols (e.g., 'N' for normal, 'A' for atrial premature beat)
```
we can then use these annotations to label data for machine learning tasks, such as classifying normal vs. arrhythmic beats.

##### Example 3: Writing the Own ECG Data
To create and write the own ECG data:
```python
import numpy as np

# Create sample ECG data (2 channels, 1000 samples)
signals = np.random.rand(1000, 2)  # Replace with actual ECG data

# Write to a WFDB record
wfdb.wrsamp('my_ecg', fs=360, units=['mV', 'mV'], sig_name=['I', 'II'], p_signal=signals, fmt=['16', '16'])
```
This creates `my_ecg.hea` and `my_ecg.dat` files, which we can read back using `wfdb.rdrecord('my_ecg')`.

##### Example 4: Visualizing Annotations on ECG
To combine signals and annotations for analysis:
```python
import matplotlib.pyplot as plt

# Read record and annotations
record = wfdb.rdrecord('100', pn_dir='path/to/mitdb')
ann = wfdb.rdann('100', 'atr', pn_dir='path/to/mitdb')

# Plot ECG with annotations
plt.figure(figsize=(10, 4))
plt.plot(record.p_signal[:, 0])
for i in range(len(ann.sample)):
    plt.axvline(x=ann.sample[i], color='r', linestyle='--', linewidth=1)
    plt.text(ann.sample[i], 1, ann.symbol[i], color='r')
plt.title('ECG Signal with Annotations - Record 100')
plt.xlabel('Samples')
plt.ylabel('Amplitude (mV)')
plt.show()
```
This visualization helps we understand the relationship between the ECG signal and annotated events, such as R-peaks or arrhythmias.

#### Resources and Further Learning
- **Official Documentation**: The [WFDB Python Documentation](https://wfdb.readthedocs.io/en/latest/) provides detailed API references and examples.
- **Demo Notebook**: Access the demo notebook on [GitHub](https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb) for practical use cases.
- **Tutorials**: Explore MIMIC WFDB Tutorials at [MIMIC WFDB Tutorials](https://wfdb.io/mimic_wfdb_tutorials/) for additional resources, though note some are still in development as of May 2025.
- **Datasets**: Use the MIT-BIH Arrhythmia Database at [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and other ECG datasets for practice.
