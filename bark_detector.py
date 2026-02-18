import sounddevice as sd
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import os
import csv
import time
from datetime import datetime
import queue

# Check for YAMNet and TFLite files
model_files_exist = os.path.exists("yamnet.tflite")
if not model_files_exist:
    print("Required file yamnet.tflite not found. See readme for more information.")
    quit()
model_files_exist = os.path.exists("yamnet_class_map.csv")
if not model_files_exist:
    print("Required file yamnet_class_map.csv not found. See readme for more information.")
    quit()

# Check for microphone access
try:
    default_input_device = sd.query_devices(kind="input")
except sd.PortAudioError:
    print("No microphones found. Exiting program.")
    quit()

# Load YAMNet class map
import csv as csvlib
with open("yamnet_class_map.csv", newline='') as f:
    reader = csvlib.reader(f)
    class_names = [row[2].strip().lower().replace('"', '') for row in reader]

# Specify dog-related labels
DOG_CLASSES = set(name.lower() for name in {
    "dog", "bark", "dog bark", "howl", "growling",
    "domestic animals, pets", "animal", "yip", "whimper", 
    "dog howling"
})

# Set confidence level required for inference [0..1]
# The higher the number, the more confidence is required to match audio sample to class
CONFIDENCE = 0.2

# Load and set up the TFLite model
interpreter = Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Audio parameters
SAMPLE_RATE = 16000 # per second
FRAME_DURATION = 0.975  # in seconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

# Real-time audio buffer
q = queue.Queue()

# Callback function for processing audio from sound device
# Called automatically when audio chunk is ready
def audio_callback(indata, frames, time_, status):
    if status:
        print(status) # Report audio stream errors to console
    mono = indata.mean(axis=1) # Downmix stereo to mono
    q.put(mono.copy()) # Add audio chunk to queue

# Classification function
# Runs the model on the given audio and returns an array of the 3 most likely categories
def classify_audio(waveform):
    # Convert waveform to tensor, match expected data shape
    tensor = np.reshape(waveform, (1, len(waveform))).astype(np.float32).flatten()
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke() # Execute inference

    # Store confidence scores for each category
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get class indices of top 3 scores (highest first)
    top_indices = np.argsort(scores)[-3:][::-1]
    return [(class_names[i], scores[i]) for i in top_indices]

# Logging setup

# Check if file exists, if not, create it and add header
file_exists = os.path.exists("bark_log.csv")

# Set to write timestamp, highest sound match, and duration of sound in seconds to a CSV file
log_file = open("bark_log.csv", mode="a", newline="")
log_writer = csv.writer(log_file)
if not file_exists:
    log_writer.writerow(["timestamp", "event", "duration"])

# Bark session state
barking = False
barking_start_time = None
last_bark_time = None
last_bark_print_time = None
BARK_TIMEOUT = 1.5  # seconds of silence to end a bark session

print(f"Listening with {default_input_device['name']}... Press Ctrl+C to stop.")

# Main program loop
# Listen to audio from microphone, pass to inference model, record dog barks to disk
try:
    with sd.InputStream(
        callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE
    ):
        while True:
            if not q.empty():
                frame = q.get()
                if len(frame) != FRAME_SIZE:
                    continue  # Skip incomplete frame

                now = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                top_classes = classify_audio(frame)

                # Optional: print top 3 classes
                #print(f"[{timestamp}] Top 3: {[(l, round(c, 2)) for l, c in top_classes]}")

                # Determine if barking detected
                bark_detected = any(
                    label.strip().lower() in DOG_CLASSES and confidence > CONFIDENCE
                    for label, confidence in top_classes
                )

                if bark_detected:
                    if not barking:
                        barking = True
                        barking_start_time = now
                        last_bark_print_time = now
                        print(f"[{timestamp}] Barking started")
                    else:
                        if now - last_bark_print_time >= 5:
                            elapsed = int(now - barking_start_time)
                            print(f"[{timestamp}] Barking for {elapsed} seconds")
                            last_bark_print_time = now
                    last_bark_time = now

                elif barking and (now - last_bark_time > BARK_TIMEOUT):
                    barking = False
                    duration = round(now - barking_start_time, 2)
                    print(f"[{timestamp}] Barking ended (duration: {duration}s)")
                    log_writer.writerow([timestamp, "barking", duration])
                    log_file.flush()
                    barking_start_time = None
                    last_bark_time = None

            else:
                time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    log_file.close()
