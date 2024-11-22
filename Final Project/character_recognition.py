from PIL import Image
import pytesseract
import cv2

# Specify the path to the Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'a.jpg'
#characters = pytesseract.image_to_boxes(image)
image_cv = cv2.imread(image_path)

# Preprocess the image
gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
denoised_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Convert the preprocessed OpenCV image to PIL format
preprocessed_image = Image.fromarray(denoised_image)

#image = Image.open(image_path)

# Perform OCR and get bounding boxes for each character
characters = pytesseract.image_to_boxes(preprocessed_image)

# Print character and its bounding box
print("Detected characters and bounding boxes:")
for char_data in characters.splitlines():
    char, x1, y1, x2, y2, _ = char_data.split()  # Character and bounding box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(f"Character: '{char}' | Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

# Convert the image to OpenCV format
image_cv = cv2.imread(image_path)
image_h, image_w, _ = image_cv.shape

# Draw bounding boxes for each detected character
for char_data in characters.splitlines():
    char, x1, y1, x2, y2, _ = char_data.split()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Adjust coordinates since Tesseract uses an inverted Y-axis
    y1, y2 = image_h - y1, image_h - y2
    cv2.rectangle(image_cv, (x1, y2), (x2, y1), (0, 255, 0), 2)
    cv2.putText(image_cv, char, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display the image with bounding boxes
cv2.imshow('Character Bounding Boxes', image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
