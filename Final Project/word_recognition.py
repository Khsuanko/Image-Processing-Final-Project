from PIL import Image
import pytesseract
import cv2

# Specify the path to the Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'a.jpg'
#image = Image.open(image_path)
image_cv = cv2.imread(image_path)

# Preprocess the image
gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
denoised_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Convert the preprocessed OpenCV image to PIL format
preprocessed_image = Image.fromarray(denoised_image)

# Get OCR data with bounding boxes
#ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
ocr_data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)

# Iterate through each word and its bounding box
for i in range(len(ocr_data['text'])):
    word = ocr_data['text'][i]
    if word.strip():
        x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                      ocr_data['width'][i], ocr_data['height'][i])
        print(f"Word: '{word}' | Bounding Box: x={x}, y={y}, w={w}, h={h}")

# Convert the image to OpenCV format
image_cv = cv2.imread(image_path)

# Draw bounding boxes
for i in range(len(ocr_data['text'])):
    word = ocr_data['text'][i]
    if word.strip():
        x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                      ocr_data['width'][i], ocr_data['height'][i])
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_cv, word, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display the image with bounding boxes
cv2.imshow('Text Detection', image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
