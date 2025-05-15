To work with ECG (electrocardiogram) data for research, especially as a beginner in signal processing with a focus on biomedical applications, understanding key terms is essential. Below, I’ll explain all the fundamental terms, focusing on ECG-related biomedical research. These terms cover ECG basics, signal processing, and research-oriented concepts, ensuring a solid foundation to process ECG data, conduct research.

### Key Terms for ECG and Signal Processing

#### 1. ECG (Electrocardiogram)
- **Definition**: A recording of the electrical activity of the heart over time, measured using electrodes placed on the skin.
- **Significance**: ECGs are used to monitor heart function, detect abnormalities (e.g., arrhythmias), and extract features like heart rate for research.
- **Example**: The wavy line you see on a heart monitor, showing peaks and valleys, is an ECG signal.

#### 2. ECG Waveform Components
These are the distinct parts of an ECG signal corresponding to different phases of the heart’s electrical cycle:
- **P Wave**: A small bump before the main spike, caused by atrial depolarization (atria contract to push blood into ventricles).
- **QRS Complex**: The largest part of the ECG, consisting of:
  - **Q Wave**: A small downward dip before the main peak.
  - **R Wave**: The tall, sharp peak (the most prominent feature, used to detect heartbeats).
  - **S Wave**: A downward dip after the R wave.
  - Represents ventricular depolarization (ventricles contract to pump blood).
- **T Wave**: A wave after the QRS complex, caused by ventricular repolarization (ventricles recover electrically).
- **Significance**: Analyzing these components helps diagnose conditions (e.g., a missing P wave may indicate atrial fibrillation).
- **Example**: In the code, `nk.ecg_delineate` identifies P, QRS, and T wave locations.

#### 3. R-Peak
- **Definition**: The highest point in the QRS complex, corresponding to the peak of ventricular depolarization.
- **Significance**: R-peaks are used to detect heartbeats and calculate heart rate or heart rate variability (HRV).
- **Example**: In the code, `nk.ecg_peaks` marks R-peaks with red dots on the ECG plot.

#### 4. R-R Interval
- **Definition**: The time (in seconds) between two consecutive R-peaks.
- **Significance**: Used to calculate heart rate (Heart Rate = 60 / R-R interval in seconds) and analyze HRV.
- **Example**: If R-peaks occur at samples 100 and 460 in a 360 Hz signal, the R-R interval is (460-100)/360 = 1 second, so heart rate = 60 BPM.

#### 5. Heart Rate
- **Definition**: The number of heartbeats per minute, measured in beats per minute (BPM).
- **Significance**: A key metric in ECG research, used to assess cardiac health (e.g., normal range: 60-100 BPM at rest).
- **Example**: In the code, `nk.ecg_rate` computes heart rate from R-R intervals.

#### 6. Heart Rate Variability (HRV)
- **Definition**: The variation in time intervals between consecutive heartbeats (R-R intervals).
- **Significance**: HRV reflects autonomic nervous system activity and is used in research to study stress, fitness, or cardiac disorders.
- **Example**: Metrics like SDNN (standard deviation of R-R intervals) or RMSSD (root mean square of successive differences) quantify HRV.

#### 7. Sampling Rate
- **Definition**: The number of times an ECG signal is measured per second, measured in Hertz (Hz).
- **Significance**: Determines the resolution of the ECG signal. Common rates are 250-500 Hz for clinical ECGs (e.g., MIT-BIH uses 360 Hz).
- **Example**: At 360 Hz, 360 samples are recorded per second, so each sample represents 1/360 = 0.00278 seconds.

#### 8. Signal
- **Definition**: A measurable quantity that varies over time, like the voltage in an ECG.
- **Significance**: ECG is a time-series signal, where voltage (in millivolts, mV) changes with each heartbeat.
- **Example**: The `ecg_signal` in the code is an array of voltage values over time.

#### 9. Noise
- **Definition**: Unwanted components in the ECG signal that obscure the true heart signal.
- **Types**:
  - **Baseline Wander**: Slow drift in the signal due to breathing or movement.
  - **Power Line Interference**: 50/60 Hz noise from electrical devices.
  - **Muscle Artifacts**: High-frequency noise from muscle movements.
- **Significance**: Noise must be removed to accurately analyze ECG features.
- **Example**: In the code, `nk.ecg_clean` removes noise like baseline wander.

#### 10. Filtering
- **Definition**: The process of removing noise from a signal to isolate the true ECG components.
- **Types**:
  - **Low-Pass Filter**: Removes high-frequency noise (e.g., muscle artifacts).
  - **High-Pass Filter**: Removes low-frequency noise (e.g., baseline wander).
  - **Band-Pass Filter**: Allows frequencies in a specific range (e.g., 0.5-40 Hz for ECG).
  - **Notch Filter**: Removes specific frequencies (e.g., 60 Hz power line noise).
- **Significance**: Filtering is critical for cleaning ECG data before analysis.
- **Example**: NeuroKit2’s `ecg_clean` applies a combination of filters.

#### 11. Feature Extraction
- **Definition**: Identifying and measuring specific characteristics of the ECG signal, such as wave amplitudes, durations, or intervals.
- **Examples**:
  - QRS duration (time from Q to S wave).
  - P-wave amplitude.
  - ST segment elevation (between S and T waves, used to detect heart attacks).
- **Significance**: Features are used in research for diagnosis, classification, or machine learning models.
- **Example**: In the code, `nk.ecg_delineate` extracts features like P and T wave locations.

#### 12. Time Domain
- **Definition**: Analyzing the ECG signal based on its amplitude over time.
- **Significance**: Most ECG features (e.g., R-peaks, heart rate) are computed in the time domain.
- **Example**: Plotting the ECG signal (as in the code) shows its time-domain representation.

#### 13. Frequency Domain
- **Definition**: Analyzing the ECG signal based on its frequency components (how fast the signal oscillates).
- **Significance**: Used to identify noise (e.g., 60 Hz interference) or study HRV.
- **Example**: Applying a Fourier Transform (not in the code) converts the signal to the frequency domain.

#### 14. Peak Detection
- **Definition**: Identifying prominent points in the ECG signal, like R-peaks.
- **Significance**: Essential for heartbeat detection and feature extraction.
- **Example**: In the code, `nk.ecg_peaks` detects R-peaks automatically.

#### 15. Baseline Wander
- **Definition**: A low-frequency drift in the ECG signal, often caused by breathing or electrode movement.
- **Significance**: Must be removed to avoid distorting wave shapes.
- **Example**: A high-pass filter in `nk.ecg_clean` corrects baseline wander.

#### 16. Artifact
- **Definition**: Any unwanted signal component, including noise or errors from equipment/patient movement.
- **Significance**: Artifacts can lead to incorrect analysis, so preprocessing is crucial.
- **Example**: Muscle tremors during ECG recording create artifacts.

#### 17. Lead
- **Definition**: A specific electrode configuration used to measure the ECG signal from different angles of the heart.
- **Types**:
  - **12-Lead ECG**: Standard clinical setup with 12 perspectives (e.g., Lead I, II, V1-V6).
  - **MLII**: A common lead in research datasets like MIT-BIH (used in the code).
- **Significance**: Different leads highlight different heart activities, affecting feature visibility.
- **Example**: The code uses the MLII lead from the MIT-BIH dataset.

#### 18. Annotation
- **Definition**: Labels or markers in ECG data indicating events like R-peaks or abnormalities (e.g., ventricular ectopic beats).
- **Significance**: Annotations in datasets like MIT-BIH help validate algorithms or train machine learning models.
- **Example**: MIT-BIH records include annotations for normal and arrhythmic beats.

#### 19. Arrhythmia
- **Definition**: An abnormal heart rhythm, such as atrial fibrillation or ventricular tachycardia.
- **Significance**: A key focus in ECG research for automated detection using signal processing or machine learning.
- **Example**: Project 2 in the previous tutorial involves detecting arrhythmias.

#### 20. Signal-to-Noise Ratio (SNR)
- **Definition**: The ratio of the desired ECG signal’s power to the noise’s power, often in decibels (dB).
- **Significance**: Higher SNR indicates a cleaner signal, critical for accurate analysis.
- **Example**: Filtering increases SNR by reducing noise.

#### 21. Preprocessing
- **Definition**: Steps to clean and prepare ECG data for analysis, including filtering and artifact removal.
- **Significance**: Ensures reliable feature extraction and algorithm performance.
- **Example**: The code’s `nk.ecg_clean` is a preprocessing step.

#### 22. Segmentation
- **Definition**: Dividing the ECG signal into smaller parts, such as individual heartbeats or QRS complexes.
- **Significance**: Simplifies analysis of specific events (e.g., comparing QRS shapes).
- **Example**: R-peak detection enables segmenting heartbeats.

#### 23. Fourier Transform
- **Definition**: A mathematical tool to convert a time-domain signal into its frequency-domain representation.
- **Significance**: Helps identify noise frequencies or analyze HRV.
- **Example**: Not used in the code but relevant for advanced filtering.

#### 24. Wavelet Transform
- **Definition**: A method to analyze the ECG signal at different time and frequency scales.
- **Significance**: Useful for detecting transient features (e.g., QRS complexes) or denoising.
- **Example**: Advanced research may use wavelets instead of traditional filters.

#### 25. Machine Learning in ECG
- **Definition**: Using algorithms to automatically classify or predict ECG features (e.g., normal vs. arrhythmic beats).
- **Terms**:
  - **Feature Vector**: A set of extracted features (e.g., QRS duration, R-R interval) used as input to a model.
  - **Classification**: Labeling ECG beats as normal or abnormal.
  - **Training Dataset**: Labeled ECG data (e.g., MIT-BIH annotations) used to teach the model.
- **Significance**: Common in modern ECG research for automated diagnosis.
- **Example**: Project 2 suggests using logistic regression for arrhythmia detection.

#### 26. Dataset
- **Definition**: A collection of ECG recordings used for research, often with annotations.
- **Examples**:
  - **MIT-BIH Arrhythmia Database**: Contains 48 ECG records with normal and arrhythmic beats (used in the code).
  - **PTB Diagnostic ECG Database**: Includes 12-lead ECGs for various conditions.
- **Significance**: Essential for developing and testing ECG algorithms.

#### 27. Ground Truth
- **Definition**: The true labels or annotations in a dataset (e.g., cardiologist-marked R-peaks).
- **Significance**: Used to evaluate the accuracy of your algorithm.
- **Example**: MIT-BIH annotations provide ground truth for R-peak detection.

#### 28. Sensitivity and Specificity
- **Definition**:
  - **Sensitivity**: The percentage of true positive detections (e.g., correctly identified R-peaks).
  - **Specificity**: The percentage of true negative detections (e.g., correctly identified non-R-peaks).
- **Significance**: Metrics to evaluate the performance of ECG algorithms.
- **Example**: If your code detects 95 out of 100 R-peaks, sensitivity is 95%.

#### 29. QRS Duration
- **Definition**: The time from the start of the Q wave to the end of the S wave, typically 80-120 ms.
- **Significance**: Prolonged QRS duration may indicate conditions like bundle branch block.
- **Example**: Extracted via `nk.ecg_delineate` in the code.

#### 30. ST Segment
- **Definition**: The flat section between the S wave and T wave.
- **Significance**: Elevation or depression in the ST segment can indicate heart attack or ischemia.
- **Example**: Advanced research may analyze ST segment changes.

