import cv2

def load_haar_cascade():
    """Load openCV's built-in face detector"""
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    return face_cascade
def detect_objects(face_cascade, gray_frame):
    """Detect faces in the given grayscale frame"""
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors = 5,
        minSize=(30, 30)
    )
    return faces