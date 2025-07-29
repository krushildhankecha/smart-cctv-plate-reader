import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text = ''
    for (bbox, txt, prob) in result:
        if prob > 0.5:
            text += txt + ' '
    return text.strip()
