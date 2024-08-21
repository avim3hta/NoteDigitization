import cv2
import pytesseract

path = "C:/Users/raceb/OneDrive/Documents/CPI/impartus/1.mp4"
# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(path)

# Set the resolution of the video capture (Optional: Adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale (improves OCR accuracy)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use thresholding to preprocess the image (optional)
    _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR on the thresholded frame
    ocr_text = pytesseract.image_to_string(thresh_frame, lang='eng')

    # Display the OCR result on the frame
    cv2.putText(frame, ocr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video with OCR', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
