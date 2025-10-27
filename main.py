import cv2
import time
import numpy as np
from numba import cuda
from preprocess_kernels import rgb_to_gray_kernel, normalize_kernel, blur_kernel
from utils import load_haar_cascade, detect_objects


# ---------- CPU reference pipeline (for comparison) ----------
def cpu_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    return (blur * 255).astype(np.uint8)


# ---------------------- MAIN PIPELINE ---------------------- #
def main():
    print("Starting camera test...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Unable to open webcam.")
        return

    print("✅ Webcam opened successfully")

    # --- Load face detection model ---
    face_cascade = load_haar_cascade()

    # --- Define Gaussian Kernel (3x3) ---
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32)
    d_kernel = cuda.to_device(gaussian_kernel)

    # --- FPS calculation setup ---
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received from webcam.")
            break

        # ---------- CPU reference blur ----------
        start_cpu = time.time()
        cpu_blurred = cpu_blur(frame)
        cpu_time = (time.time() - start_cpu) * 1000  # ms

        # ---------- GPU (CUDA) pipeline ----------
        start_gpu = time.time()

        d_input = cuda.to_device(frame)
        d_gray = cuda.device_array((frame.shape[0], frame.shape[1]), dtype=np.float32)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(frame.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(frame.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        rgb_to_gray_kernel[blockspergrid, threadsperblock](d_input, d_gray)
        d_norm = cuda.device_array_like(d_gray)
        normalize_kernel[blockspergrid, threadsperblock](d_gray, d_norm)
        d_blur = cuda.device_array_like(d_norm)
        blur_kernel[blockspergrid, threadsperblock](d_norm, d_blur, d_kernel)
        blur_frame = d_blur.copy_to_host()
        disp_frame = (blur_frame * 255).astype(np.uint8)

        gpu_time = (time.time() - start_gpu) * 1000  # ms

        # ---------- Face detection ----------
        gray_for_detection = disp_frame
        faces = detect_objects(face_cascade, gray_for_detection)
        for (x, y, w, h) in faces:
            cv2.rectangle(disp_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ---------- FPS + timing display ----------
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(disp_frame, f"GPU FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp_frame, f"CPU Blur: {cpu_time:.2f} ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(disp_frame, f"GPU Blur: {gpu_time:.2f} ms", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ---------- Display both feeds ----------
        cv2.imshow("Blurred (CUDA + Detection)", disp_frame)
        cv2.imshow("Raw Webcam Feed", frame)

        # Exit when 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
