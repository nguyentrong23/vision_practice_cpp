import cv2
  
# path
path = "template.bmp"
  
# Reading an image in default mode
src = cv2.imread(path)
  
# Use Flip code 0 to flip vertically
image = cv2.flip(src, 0)

cv2.imwrite("template-1.bmp", image)