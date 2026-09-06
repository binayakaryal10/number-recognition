"""
Real-Time Camera Deployment for CNN Handwritten Digit Recognition
Uses live webcam feed to recognize handwritten numbers with high accuracy.
Supports:
- Interactive Target Box (ROI) mode with live probability distribution
- Multi-digit Auto-Contour detection mode
- Preprocessed 28x28 preview (displays what the CNN actually sees)
- Keyboard controls: 'q'=Quit, 'm'=Toggle Mode, 't'=Toggle Threshold, 'p'=Freeze, 's'=Snapshot
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf

from preprocess_utils import preprocess_digit, find_digit_regions

DEFAULT_MODEL_PATH = "mnist_cnn_model.keras"

def draw_probability_bars(frame, probs, top_left=(20, 160), bar_width=180, bar_height=14):
    """Draws a neat mini horizontal bar chart of probabilities for digits 0 to 9."""
    x0, y0 = top_left
    pred_idx = np.argmax(probs)
    
    # Semi-transparent background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 25), (x0 + bar_width + 80, y0 + 10 * 20 + 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "Probabilities (0-9):", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    for i in range(10):
        y = y0 + i * 20
        p = float(probs[i])
        fill_w = int(p * bar_width)
        
        # Color: bright green for highest probability, light gray for others
        color = (0, 255, 120) if i == pred_idx and p > 0.3 else (140, 140, 140)
        
        # Digit label
        cv2.putText(frame, f"{i}:", (x0, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Bar background
        cv2.rectangle(frame, (x0 + 22, y), (x0 + 22 + bar_width, y + bar_height), (50, 50, 50), -1)
        # Filled bar
        if fill_w > 0:
            cv2.rectangle(frame, (x0 + 22, y), (x0 + 22 + fill_w, y + bar_height), color, -1)
        # Percent text
        cv2.putText(frame, f"{p*100:4.1f}%", (x0 + 28 + bar_width, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

def draw_hud(frame, mode, fps, is_frozen, show_thresh):
    """Draws top information banner and keyboard control hints."""
    h, w = frame.shape[:2]
    
    # Top HUD Bar
    cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 15), -1)
    
    title = "CNN Real-Time Digit Recognizer"
    mode_str = f"Mode: {'[1] Center ROI' if mode == 1 else '[2] Auto-Detect Contours'}"
    fps_str = f"FPS: {fps:4.1f}"
    
    cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, mode_str, (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_str, (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Bottom Controls Bar
    cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 15), -1)
    controls = "[Q] Quit  |  [M] Toggle Mode  |  [T] Toggle Threshold  |  [P] Freeze  |  [S] Snapshot"
    if is_frozen:
        controls += "  -- [PAUSED] --"
    cv2.putText(frame, controls, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

def run_detector(model_path=DEFAULT_MODEL_PATH, camera_index=0):
    # 1. Load trained CNN model
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found at: {os.path.abspath(model_path)}")
        print("Please train the model first by running: python train_model.py\n")
        return

    print(f"Loading CNN model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # 2. Open Video Capture
    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {camera_index}.")
        print("Tip: If using an external webcam, try: python realtime_detector.py --camera 1")
        return

    # Attempt to set comfortable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Handwritten Digit Recognizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Handwritten Digit Recognizer", 1100, 700)

    # State variables
    mode = 1           # 1 = Target ROI, 2 = Auto-Contour Detection
    show_thresh = False
    is_frozen = False
    frozen_frame = None
    snapshot_count = 0
    prev_time = time.time()
    fps = 0.0

    # Default ROI Box dimensions
    roi_size = 240

    print("\n" + "=" * 60)
    print(" Camera Detection Running!")
    print(" Controls:")
    print("   'q' or ESC : Exit")
    print("   'm'        : Toggle between Center ROI and Auto-Detect Mode")
    print("   't'        : Toggle Thresholded View on/off")
    print("   'p'        : Freeze / Unfreeze live camera feed")
    print("   's'        : Save current snapshot to disk")
    print("=" * 60 + "\n")

    while True:
        if not is_frozen:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera.")
                time.sleep(0.05)
                continue
            # Mirror frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
        else:
            frame = frozen_frame.copy()

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        h, w = frame.shape[:2]
        display_frame = frame.copy()

        # =====================================================================
        # MODE 1: Interactive Center ROI Mode
        # =====================================================================
        if mode == 1:
            rx1 = (w - roi_size) // 2
            ry1 = (h - roi_size) // 2
            rx2 = rx1 + roi_size
            ry2 = ry1 + roi_size

            # Extract ROI crop
            roi_crop = frame[ry1:ry2, rx1:rx2]
            
            # Preprocess to MNIST standard
            input_tensor, preprocessed_28, is_empty = preprocess_digit(roi_crop)

            if not is_empty:
                probs = model.predict(input_tensor, verbose=0)[0]
                pred_digit = int(np.argmax(probs))
                confidence = float(probs[pred_digit]) * 100.0

                # Highlight ROI box (Bright Green if confident, Orange otherwise)
                box_color = (0, 255, 0) if confidence > 65 else (0, 165, 255)
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), box_color, 3)

                # Corner brackets
                bracket_len = 25
                cv2.line(display_frame, (rx1, ry1), (rx1 + bracket_len, ry1), box_color, 4)
                cv2.line(display_frame, (rx1, ry1), (rx1, ry1 + bracket_len), box_color, 4)
                cv2.line(display_frame, (rx2, ry1), (rx2 - bracket_len, ry1), box_color, 4)
                cv2.line(display_frame, (rx2, ry1), (rx2, ry1 + bracket_len), box_color, 4)
                cv2.line(display_frame, (rx1, ry2), (rx1 + bracket_len, ry2), box_color, 4)
                cv2.line(display_frame, (rx1, ry2), (rx1, ry2 - bracket_len), box_color, 4)
                cv2.line(display_frame, (rx2, ry2), (rx2 - bracket_len, ry2), box_color, 4)
                cv2.line(display_frame, (rx2, ry2), (rx2, ry2 - bracket_len), box_color, 4)

                # Display large prediction result above ROI
                res_text = f"Digit: {pred_digit}  ({confidence:.1f}%)"
                (tw, th), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
                cv2.rectangle(display_frame, (rx1, ry1 - th - 20), (rx1 + tw + 20, ry1 - 5), (20, 20, 20), -1)
                cv2.putText(display_frame, res_text, (rx1 + 10, ry1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, box_color, 2, cv2.LINE_AA)

                # Draw probability bar graph
                draw_probability_bars(display_frame, probs, top_left=(25, 80))
            else:
                # Idle state: waiting for digit inside ROI
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (180, 180, 180), 2)
                guide_text = "Place digit here"
                (tw, th), _ = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                cv2.putText(display_frame, guide_text, (rx1 + (roi_size - tw) // 2, ry1 + (roi_size // 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

            # --- PiP: Magnified Preprocessed 28x28 preview ---
            # Shows user the exact normalized view fed into the CNN
            preview_size = 112
            pip_x = w - preview_size - 25
            pip_y = 65
            preview_bgr = cv2.cvtColor(cv2.resize(preprocessed_28, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(display_frame, (pip_x - 3, pip_y - 20), (pip_x + preview_size + 3, pip_y + preview_size + 3), (40, 40, 40), -1)
            cv2.putText(display_frame, "CNN Input (28x28):", (pip_x, pip_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            display_frame[pip_y:pip_y+preview_size, pip_x:pip_x+preview_size] = preview_bgr
            cv2.rectangle(display_frame, (pip_x, pip_y), (pip_x + preview_size, pip_y + preview_size), (0, 255, 255), 1)

        # =====================================================================
        # MODE 2: Multi-Digit Auto-Contour Detection Mode
        # =====================================================================
        elif mode == 2:
            digit_boxes, thresh_img = find_digit_regions(frame)
            
            for (bx, by, bw, bh) in digit_boxes:
                crop = frame[by:by+bh, bx:bx+bw]
                input_tensor, _, is_empty = preprocess_digit(crop)
                
                if not is_empty:
                    probs = model.predict(input_tensor, verbose=0)[0]
                    pred_digit = int(np.argmax(probs))
                    confidence = float(probs[pred_digit]) * 100.0
                    
                    if confidence > 45:
                        color = (0, 255, 0) if confidence > 70 else (0, 200, 255)
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                        
                        label = f"{pred_digit} ({confidence:.0f}%)"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (bx, by - lh - 8), (bx + lw + 6, by), (20, 20, 20), -1)
                        cv2.putText(display_frame, label, (bx + 3, by - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            if show_thresh:
                # Display thresholded binary mask
                thresh_bgr = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
                display_frame = cv2.addWeighted(display_frame, 0.6, thresh_bgr, 0.4, 0)

        # Draw HUD overlays
        draw_hud(display_frame, mode, fps, is_frozen, show_thresh)

        # Show frame
        cv2.imshow("Handwritten Digit Recognizer", display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('m'):
            mode = 2 if mode == 1 else 1
            print(f"Switched to Mode: {'[1] Center ROI' if mode == 1 else '[2] Auto-Contour'}")
        elif key == ord('t'):
            show_thresh = not show_thresh
            print(f"Threshold overlay: {'ON' if show_thresh else 'OFF'}")
        elif key == ord('p'):
            is_frozen = not is_frozen
            if is_frozen:
                frozen_frame = frame.copy()
                print("Live feed PAUSED. Press 'p' again to resume.")
            else:
                print("Live feed RESUMED.")
        elif key == ord('s'):
            snapshot_count += 1
            filename = f"digit_snapshot_{snapshot_count}.png"
            cv2.imwrite(filename, display_frame)
            print(f"Saved snapshot to {os.path.abspath(filename)}")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time handwritten digit recognition using CNN")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained .keras model file")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()
    
    run_detector(model_path=args.model, camera_index=args.camera)
