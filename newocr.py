import cv2
import pytesseract
import numpy as np

# Initialize Tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(frame):
    """Preprocess the image to improve OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better results in varying lighting conditions
    processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Optional: Apply a slight blur to reduce noise (Gaussian Blur)
    processed_image = cv2.GaussianBlur(processed_image, (3, 3), 0)

    # Optional: Morphological transformations to enhance text structure
    kernel = np.ones((2, 2), np.uint8)
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

    return processed_image

def extract_text_from_image(frame):
    """Extract text from a preprocessed frame."""
    # Preprocess the frame to improve OCR accuracy
    processed_image = preprocess_image(frame)

    # Perform OCR on the processed image
    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')

    return text

def main():
    video_path = 'C:/Users/raceb/OneDrive/Documents/CPI/impartus/1.mp4'
    interval = 10  # Seconds

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Capture frame at the specified interval
        if count % int(fps * interval) == 0:
            frames.append(frame)

        count += 1

    cap.release()

    # Perform OCR on extracted frames
    for idx, frame in enumerate(frames):
        text = extract_text_from_image(frame)
        print(f"Text from Frame {idx + 1}:\n{text}\n{'-' * 40}")

if __name__ == "__main__":
    main()
