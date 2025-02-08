import cv2
import os
import numpy as np


def get_video_resolution(video_path):
    """
    Checks and returns the resolution of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def add_noise_and_artifacts(frame, noise_std=25, compression_quality=30):
    """
    Adds Gaussian noise and then JPEG compression artifacts to an image (NumPy array).
    1) noise_std controls the strength of Gaussian noise.
    2) compression_quality sets JPEG encoding quality (lower = more artifacts).
    """
    # 1) Convert to float and add Gaussian noise
    noisy_frame = frame.astype(np.float32)
    noise = np.random.normal(0, noise_std, frame.shape).astype(np.float32)
    noisy_frame += noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)

    # 2) Add JPEG compression artifacts by in-memory encode/decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    success, enc_img = cv2.imencode('.jpg', noisy_frame, encode_param)
    if not success:
        # If encoding fails, just return the noisy frame
        return noisy_frame

    # Decode it back to simulate compression artifacts
    decoded = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
    return decoded


def extract_seven_frame_sequences(video_path, output_dir, noisy_output_dir):
    """
    Extracts consecutive 7-frame sequences from `video_path` and saves them 
    as PNGs in subfolders of both `output_dir` (clean) and `noisy_output_dir`.
    The 'noisy' frames will have Gaussian noise + JPEG compression artifacts.
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frames_per_sequence = 7
    seq_index = 0
    frame_buffer = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        frame_count += 1
        frame_buffer.append(frame)

        # Once we have 7 frames, save them
        if len(frame_buffer) == frames_per_sequence:
            seq_index += 1

            # Create matching subfolders for clean and noisy
            seq_folder_clean = os.path.join(output_dir, f"clip_{seq_index:05d}")
            seq_folder_noisy = os.path.join(noisy_output_dir, f"clip_{seq_index:05d}")
            os.makedirs(seq_folder_clean, exist_ok=True)
            os.makedirs(seq_folder_noisy, exist_ok=True)

            # Save both clean and noisy frames
            for i, f in enumerate(frame_buffer, start=1):
                # Save clean frame
                clean_name = f"im{i}.png"
                clean_path = os.path.join(seq_folder_clean, clean_name)
                cv2.imwrite(clean_path, f)

                # Generate and save noisy+artifacted frame
                degraded = add_noise_and_artifacts(f, 
                                                   noise_std=25, 
                                                   compression_quality=30)
                degraded_name = f"im{i}.png"
                degraded_path = os.path.join(seq_folder_noisy, degraded_name)
                cv2.imwrite(degraded_path, degraded)

            # Clear buffer for the next sequence
            frame_buffer.clear()

    cap.release()
    print(f"Total frames read: {frame_count}")
    print(f"Total 7-frame sequences saved: {seq_index}")


if __name__ == "__main__":
    # Example usage:
    video_path = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/Timeline 2.avi"  # Replace with your 1-minute video path

    # Check video resolution
    resolution = get_video_resolution(video_path)
    if resolution is None:
        print("Error: Unable to determine video resolution. Exiting...")
    else:
        width, height = resolution
        print(f"Video resolution: {width}x{height}")

        if width != 3840 or height != 2160:
            print("Warning: Video is not in 4K resolution (3840x2160). Proceeding anyway...")
        
        # Folder for "clean" 7-frame sequences
        output_dir = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/datasets/custom_dataset/Target"

        # Folder for "noisy+artifacted" 7-frame sequences
        noisy_output_dir = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/datasets/custom_dataset/Input"

        extract_seven_frame_sequences(video_path, output_dir, noisy_output_dir)
        print("Done!")
