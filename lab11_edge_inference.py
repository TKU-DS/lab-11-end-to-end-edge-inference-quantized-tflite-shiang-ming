import cv2
import numpy as np
import time
import os
import urllib.request
import zipfile

try:
    import tflite_runtime.interpreter as tflite
    print("[*] Successfully loaded tflite_runtime.")
except ImportError:
    print("[!] tflite_runtime not found. Falling back to full tensorflow.lite...")
    import tensorflow.lite as tflite


def download_assets():
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
    img_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"

    img_name = "test_input.jpg"
    model_zip = "mobilenet_quant.zip"
    model_name = "mobilenet_v1_1.0_224_quant.tflite"
    label_name = "labels_mobilenet_quant_v1_224.txt"

    if not os.path.exists(img_name):
        print("[*] Downloading test image...")
        urllib.request.urlretrieve(img_url, img_name)

    if not os.path.exists(model_name):
        print("[*] Downloading INT8 Quantized MobileNet model from Google...")
        urllib.request.urlretrieve(model_url, model_zip)
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(model_zip)

    return img_name, model_name, label_name


def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    print("=== Week 11: End-to-End INT8 Edge Inference ===\n")

    img_path, model_path, label_path = download_assets()
    labels = load_labels(label_path)

    # TODO 1: Initialize TFLite Interpreter
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[*] Model Loaded: {model_path}")

    # =========================================================
    # PHASE 1: PRE-PROCESSING
    # =========================================================
    t0 = time.perf_counter()

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to load image.")

    # TODO 2: NumPy Pre-processing
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)

    t_pre = (time.perf_counter() - t0) * 1000

    # =========================================================
    # PHASE 2: INFERENCE
    # =========================================================
    t1 = time.perf_counter()

    # TODO 3: Execute Inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    t_inf = (time.perf_counter() - t1) * 1000

    # =========================================================
    # PHASE 3: POST-PROCESSING
    # =========================================================
    predictions = np.squeeze(output_data)
    top_1_index = np.argmax(predictions)
    confidence = predictions[top_1_index]

    print("\n[+] Inference Results:")
    print(f"    - Prediction: {labels[top_1_index]}")
    print(f"    - Quantized Confidence: {confidence} / 255")

    print("\n[+] Latency Breakdown (Virtual Machine):")
    print(f"    - Pre-processing Time: {t_pre:.2f} ms")
    print(f"    - Inference Time:      {t_inf:.2f} ms")
    print(f"    - Total Pipeline Time: {(t_pre + t_inf):.2f} ms\n")