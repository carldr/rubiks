import cv2


class Face:
  def __init__(self, squares):
    self.squares = squares

    self.expected_hues = {
      "blue": 98,
      "white": 56,
      "yellow": 40,
      "orange": 14,
      "red": 35,
      "green": 75
    }

    self.standardised_colours = {
      "blue": (255, 0, 0),
      "white": (255, 255, 255),
      "yellow": (0, 255, 255),
      "orange": (0, 165, 255),
      "red": (0, 0, 255),
      "green": (0, 255, 0)
    }

  def determine_colours(self, frame):
    print("Detecting colours")

    colours = []

    for square in self.squares:
      x, y, w, h = cv2.boundingRect(square)

      img = frame[y:y + h, x:x + w]
      hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

      mean_h, mean_s, mean_v, something = cv2.mean(hsvFrame)

      matches = sorted(self.expected_hues.items(),
                       key=lambda item: abs(mean_h - item[1]))
      match = matches[0]  # Match is a tuple (colour, hue)

      print("    Detected hue: {}, Closest match: {}".format(
        mean_h, match[0]))

      colours.append(self.standardised_colours[match[0]])

    return colours
