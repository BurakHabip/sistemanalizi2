import cv2
import pytesseract
import imutils
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def plaka_tanima(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plaka = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plaka = gray[y:y+h, x:x+w]
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            break

    if plaka is not None:
        text = pytesseract.image_to_string(plaka, config='--psm 8')
        print(f"{os.path.basename(image_path)} - Tespit Edilen Plaka: {text.strip()}")
        cv2.imshow("Plaka", plaka)
    else:
        print(f"{os.path.basename(image_path)} - Plaka bulunamadı.")

    cv2.imshow("Görüntü", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Klasördeki tüm .jpg dosyalarını alır
klasor_yolu = "plakalar"
for dosya in os.listdir(klasor_yolu):
    if dosya.lower().endswith(".jpg") or dosya.lower().endswith(".png"):
        tam_yol = os.path.join(klasor_yolu, dosya)
        plaka_tanima(tam_yol)