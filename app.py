import cv2
from face import Face


class App:
  def __init__(self):
    self.cam = cv2.VideoCapture(1)
    self.frame = None
    self.last_matched_colours = None

  def draw_points(self, points, colour):
    for point in points:
      # 45 = expected size of squares / 2
      cv2.circle(self.frame, (point[0], point[1]), 40, colour, 2)

  def draw_contours(self, contours, colour, thickness=1):
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      ratio = w / h

      # print("x: {}, y: {}, w: {}, h: {}, ratio: {}".format(x, y, w, h, ratio))

      cv2.rectangle(self.frame, (x, y), (x + w, y + h), colour, thickness)

  def find_contours(self, dilated_frame):
    target_size = 110
    target_size_deviation = 30

    contours, _ = cv2.findContours(
        dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    #  Find large, square contours
    for contour in contours:
      _x, _y, w, h = cv2.boundingRect(contour)
      ratio = w / h

      if ratio > 0.9 and ratio < 1.1 and w > target_size - target_size_deviation and w < target_size + target_size_deviation:
        filtered_contours.append(contour)

    self.draw_contours(filtered_contours, (0, 255, 0), 2)  # Bright green

    return filtered_contours

  def sort_square_contours(self, squares):
    sorted_squares_top_to_bottom = sorted(
      squares, key=lambda item: cv2.boundingRect(item)[1]
    )
    sorted_squares_top_row = sorted(
      sorted_squares_top_to_bottom[:3], key=lambda item: cv2.boundingRect(item)[0])
    sorted_squares_middle_row = sorted(
      sorted_squares_top_to_bottom[3:6], key=lambda item: cv2.boundingRect(item)[0])
    sorted_squares_bottom_row = sorted(
      sorted_squares_top_to_bottom[6:], key=lambda item: cv2.boundingRect(item)[0])

    return sorted_squares_top_row + sorted_squares_middle_row + sorted_squares_bottom_row

  def find_cube_squares(self, filtered_contours):
    # Find contours which have neighbours nearby
    for contour in filtered_contours:
      x, y, w, h = cv2.boundingRect(contour)
      x_center = x + w / 2
      y_center = y + h / 2
      size = (w + h) / 2
      distance = size * 1.3  # How far away the neighbours can be

      neighbour_positions = [
        #  Top left
        [int(x_center - distance), int(y_center - distance)],

        # Top
        [int(x_center), int(y_center - distance)],

        # Top right
        [int(x_center + distance), int(y_center - distance)],

        # Left
        [int(x_center - distance), int(y_center)],

        # Center
        [int(x_center), int(y_center)],

        # Right
        [int(x_center + distance), int(y_center)],

        # Bottom right
        [int(x_center + distance), int(y_center + distance)],

        # Bottom
        [int(x_center), int(y_center + distance)],

        # Bottom left
        [int(x_center - distance), int(y_center + distance)]
      ]

      found_squares = []

      for candidate_neighbour in filtered_contours:
        for (nx, ny) in neighbour_positions:
          cx, cy, cw, ch = cv2.boundingRect(candidate_neighbour)

          #  if the neighbour position is within the bounding box of the candidate neighbourm we count it
          if cx < nx and cy < ny and cx + cw > nx and cy + ch > ny:
            # print("Found neighbour ")
            found_squares.append(candidate_neighbour)

      if len(found_squares) == 9:
        self.draw_points(neighbour_positions, (128, 0, 0))  # Dark blue
        self.draw_contours(found_squares, (0, 0, 128), 2)    # Dark Red

        return found_squares

  def detect(self):
    grayFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.blur(grayFrame, (3, 3))
    cannyFrame = cv2.Canny(blurredFrame, 30, 60, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilatedFrame = cv2.dilate(cannyFrame, kernel)

    contours = self.find_contours(dilatedFrame)
    found_squares = self.find_cube_squares(contours)
    if found_squares:
      self.draw_contours(found_squares, (0, 0, 255), 2)    # Red
      sorted_squares = self.sort_square_contours(found_squares)

      face = Face(sorted_squares)
      last_colours = face.determine_colours(self.frame)

      #  Store the found colours/squares
      self.last_matched_colours = list(zip(sorted_squares, last_colours))

    if self.last_matched_colours:
      for square, colour in self.last_matched_colours:
        # square = self.last_matched_colours[index][0]
        # colour = self.last_matched_colours[index][1]

        x, y, w, h = cv2.boundingRect(square)
        cv2.rectangle(self.frame, (x, y), (x + w, y + h), colour, -1)

    cv2.imshow('capture', self.frame)

  def run(self):
    while True:
      ret, frame = self.cam.read()
      self.frame = frame
      self.detect()

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    self.cam.release()
    cv2.destroyAllWindows()


app = App()
