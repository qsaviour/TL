from tool.processor import processor
import time
import cv2

print('start')
t1 = time.time()
for _ in range(32):
    img = cv2.imread('tesla_000001.jpg')
    processor.augment(img)
print(time.time() - t1)
