#pip install numpy opencv-python --user

import numpy as np
import os
import sys
import cv2

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

"""
Check if a file exists
"""
def checkIfFileExists(filename):
    return os.path.isfile(filename) 

def resize_with_padding(image, width, height, DO_PADDING=True, TINY_FLOAT=1e-5):
    """
    Resizes an image to the specified size,
    adding padding to preserve the aspect ratio.
    """
    shape_out=(height,width)
    if image.ndim == 3 and len(shape_out) == 2:
        shape_out = [*shape_out, 3]
    hw_out, hw_image = [np.array(x[:2]) for x in (shape_out, image.shape)]
    resize_ratio = np.min(hw_out / hw_image)
    hw_wk = (hw_image * resize_ratio + TINY_FLOAT).astype(int)

    # Resize the image
    resized_image = cv2.resize(image, tuple(hw_wk[::-1]), interpolation=cv2.INTER_CUBIC) #INTER_NEAREST

    if not DO_PADDING or np.all(hw_out == hw_wk):
        return resized_image

    # Create a black image with the target size
    padded_image = np.zeros(shape_out, dtype=np.uint8)
    
    # Calculate the number of rows/columns to add as padding
    dh, dw = (hw_out - hw_wk) // 2
    # Add the resized image to the padded image, with padding on the left and right sides
    padded_image[dh : hw_wk[0] + dh, dw : hw_wk[1] + dw] = resized_image

    return padded_image


class FolderStreamer:
    def __init__(self, path="./", label="", width=0, height=0):
        self.path = path
        self.label = label
        self.width = width
        self.height = height
        self.should_stop = False
        self.frameNumber = 0
        self.img_files = []

        # Collect all .jpg and .png files in path
        all_files = os.listdir(self.path)
        self.img_files = sorted(
            [f for f in all_files if f.lower().endswith((".jpg", ".png"))]
        )

        if not self.img_files:
            print(f"No .jpg or .png files found in {self.path}")
            self.should_stop = True
        else:
            print(f"Found {len(self.img_files)} image files in {self.path}")

    def isOpened(self):
        return not self.should_stop

    def release(self):
        print("Release Called")
        self.should_stop = True

    def read(self):
        if self.should_stop or self.frameNumber >= len(self.img_files):
            self.should_stop = True
            return False, None

        filename = os.path.join(self.path, self.img_files[self.frameNumber])
        img = cv2.imread(filename)

        if img is None:
            print(f"Could not read image: {filename}")
            self.should_stop = True
            return False, None

        if (self.width != 0) and (self.height != 0):
            width_before = img.shape[1]
            height_before = img.shape[0]
            #print(f"Received Image size was {width_before}x{height_before}", end="")

            img = resize_with_padding(img, self.width, self.height)

            width_after  = img.shape[1]
            height_after = img.shape[0]
            #print(f" Rescaled Image size is {width_after}x{height_after}")

        self.frameNumber += 1
        return True, img

    def visualize(
                self,
                windowname='Object Detection',
                width=800,
                height=600
               ): 
            # Display output
            cv2.imshow(windowname, cv2.resize(self.img,(width,height)))

            if cv2.waitKey(10) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                self.should_stop = True
                #break

