import cv2

class App:
  def __init__(self):
    self.cam = cv2.VideoCapture(1)

  def run(self):
    while True:
      ret, frame = self.cam.read()
      cv2.imshow('Webcam', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    self.cam.release()
    cv2.destroyAllWindows()

app = App()
