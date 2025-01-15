import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance



"""
To correctly start this program you have to be in a directory ./cv_sems_DSBA
and type in a console:
python seminars/first_seminar_14.01.25/CV_seminar1.py
"""

# 2nd task
def load_display_OpenCV(photo_path):
    img = cv2.imread(photo_path)

    cv2.imshow('sample image', img)
    cv2.waitKey(0)  # waiting for a key to be pressed (required element for correct display)
    cv2.destroyAllWindows()


def load_display_Pillow(photo_path):
    image_val = Image.open(photo_path)
    image_val.show()

# 3rd task
def save_photo_OpenCV(photo_path, saving_path):
    img = cv2.imread(photo_path)
    
    # Сhoose which format you want to save the file to (we need .png)
    saving_path = os.path.join(saving_path, 'CV2_image.png')

    # save_path, img (read by cv2), params !!OPTIONAL!! (such as compression [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite(saving_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite(saving_path, img) # alternative


def save_photo_Pillow(photo_path, saving_path):
    image = Image.open(photo_path)

    # Конвертируем RGBA в RGB (на случай если у фотографии вашей есть такой параметр)
    image = image.convert('RGB')

    saving_path = os.path.join(saving_path, 'PILLOW_image.jpeg')
    image.save(saving_path)

# 4th task
def image_chennels_OpenCV(photo_path, saving_path):
    image = cv2.imread(photo_path)

    b, g, r = cv2.split(image) # cv2 has BGR (not RGB)

    cv2.imwrite(os.path.join(saving_path, 'CV2channel_r.jpg'), r)
    cv2.imwrite(os.path.join(saving_path, 'CV2channel_g.jpg'), g)
    cv2.imwrite(os.path.join(saving_path, 'CV2channel_b.jpg'), b)

    # ADDITIONAL Merge the channels back into a single image
    merged_image = cv2.merge((b, g, r))
    cv2.imwrite(os.path.join(saving_path, 'CV2merged_image.jpg'), merged_image)


def image_channels_pillow(photo_path, saving_path):
    image = Image.open(photo_path)

    # Check the number of channels in the image
    if image.mode == 'RGBA':
        # Split the image into R, G, B, A channels
        r, g, b, a = image.split()
        r.save(os.path.join(saving_path, 'Pillowchannel_r.jpg'))
        g.save(os.path.join(saving_path, 'Pillowchannel_g.jpg'))
        b.save(os.path.join(saving_path, 'Pillowchannel_b.jpg'))
        a.save(os.path.join(saving_path, 'Pillowchannel_a.jpg'))  # Save alpha channel if needed

        # Merge the channels back into a single image (without alpha)
        merged_image = Image.merge('RGB', (r, g, b))

    elif image.mode == 'RGB':
        r, g, b = image.split()
        r.save(os.path.join(saving_path, 'Pillowchannel_r.jpg'))
        g.save(os.path.join(saving_path, 'Pillowchannel_g.jpg'))
        b.save(os.path.join(saving_path, 'Pillowchannel_b.jpg'))

        merged_image = Image.merge('RGB', (r, g, b))
    else:
        raise ValueError("Unsupported image mode: {}".format(image.mode))

    # Save the merged image
    merged_image.save(os.path.join(saving_path, 'Pillowmerged_image.jpg'))

# 5th task
def resize_and_crop_opencv(photo_path, saving_path):
    image = cv2.imread(photo_path)

    resized_image = cv2.resize(image, (300, 300))

    # Crop a central region of size 200x200 pixels (In cv2 we have x,y,w,h where x,y top left coordinate)
    height, width = resized_image.shape[:2]
    start_x = (width - 200) // 2
    start_y = (height - 200) // 2
    # While croping we put starting point and adding width or height to our param (y == add height/ x == add width)
    cropped_image = resized_image[start_y:start_y + 200, start_x:start_x + 200]

    # Save the resulting images
    cv2.imwrite(os.path.join(saving_path, 'opencv_resized_image.jpg'), resized_image)
    cv2.imwrite(os.path.join(saving_path, 'opencv_cropped_image.jpg'), cropped_image)


def resize_and_crop_pillow(photo_path, saving_path):
    image = Image.open(photo_path)
    resized_image = image.resize((300, 300))

    # Convert to RGB if the image has an alpha channel
    if resized_image.mode == 'RGBA':
        resized_image = resized_image.convert('RGB')

    width, height = resized_image.size
    start_x = (width - 200) // 2
    start_y = (height - 200) // 2

    cropped_image = resized_image.crop((start_x, start_y, start_x + 200, start_y + 200))

    # Save the resulting images
    resized_image.save(os.path.join(saving_path, 'pillow_resized_image.jpg'))
    cropped_image.save(os.path.join(saving_path, 'pillow_cropped_image.jpg'))

# 6th task
def rotate_image_opencv(photo_path, saving_path):
    image = cv2.imread(photo_path)

    # Get the image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Rotate clockwise
    M_clockwise = cv2.getRotationMatrix2D(center, -45, 1.0)  # Negative angle for clockwise
    rotated_clockwise = cv2.warpAffine(image, M_clockwise, (width, height))

    # Rotate counterclockwise
    M_counterclockwise = cv2.getRotationMatrix2D(center, 45, 1.0)  # Positive angle for counterclockwise
    rotated_counterclockwise = cv2.warpAffine(image, M_counterclockwise, (width, height))

    # Save the resulting images
    cv2.imwrite(os.path.join(saving_path, 'opencv_rotated_clockwise.jpg'), rotated_clockwise)
    cv2.imwrite(os.path.join(saving_path, 'opencv_rotated_counterclockwise.jpg'), rotated_counterclockwise)


def rotate_image_pillow(photo_path, saving_path):
    image = Image.open(photo_path)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Rotate clockwise
    rotated_clockwise = image.rotate(-45, expand=True)  # Negative angle for clockwise
    # Rotate counterclockwise
    rotated_counterclockwise = image.rotate(45, expand=True)  # Positive angle for counterclockwise

    # Save the resulting images
    rotated_clockwise.save(os.path.join(saving_path, 'pillow_rotated_clockwise.jpg'))
    rotated_counterclockwise.save(os.path.join(saving_path, 'pillow_rotated_counterclockwise.jpg'))

# 7th task
def adjust_contrast_brightness_opencv(photo_path, saving_path):
    image = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel
    if image.shape[2] == 4:  # RGBA
        b, g, r, a = cv2.split(image)
        rgb_image = cv2.merge((b, g, r))  # Merge RGB channels for processing
    else:
        rgb_image = image

    # Increase contrast by 50% (1.5) and brightness by 30 units
    alpha = 1.5  # Contrast   (1.0-3.0)
    beta = 30    # Brightness (0-100)

    # Adjust contrast and brightness
    adjusted_image = cv2.convertScaleAbs(rgb_image, alpha=alpha, beta=beta)

    # If the original image had an alpha channel, add it back
    if image.shape[2] == 4:
        adjusted_image = cv2.merge((adjusted_image, a))

    # Save the modified image
    cv2.imwrite(os.path.join(saving_path, 'opencv_adjusted_image.png'), adjusted_image)


def adjust_contrast_brightness_pillow(photo_path, saving_path):
    image = Image.open(photo_path).convert('RGBA')  # Ensure it's in RGBA format (optional)

    # Increase contrast by 50%
    enhancer_contrast = ImageEnhance.Contrast(image)
    adjusted_image = enhancer_contrast.enhance(1.5)

    # Increase brightness by 30 units
    enhancer_brightness = ImageEnhance.Brightness(adjusted_image)
    adjusted_image = enhancer_brightness.enhance(1 + (30 / 255))  # Scale brightness

    adjusted_image.save(os.path.join(saving_path, 'pillow_adjusted_image.png'))

# 8th task
def histogram_opencv(photo_path, saving_path):
    image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.plot(histogram)
    plt.xlim([0, 256])

    plt.savefig(os.path.join(saving_path, 'opencv_histogram.png'))
    plt.close()


def histogram_pillow(photo_path, saving_path):
    image = Image.open(photo_path).convert('L')  # Convert to grayscale

    histogram = image.histogram()

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.bar(range(256), histogram, width=1, color='black')

    plt.savefig(os.path.join(saving_path, 'pillow_histogram.png'))
    plt.close()

# 10th task
def convert_opencv_to_pillow(photo_path, saving_path):
    image_cv = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)

    if image_cv.shape[2] == 4:  # If the image has an alpha channel
        image_rgba = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGBA)
    else:
        image_rgba = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Convert to Pillow format
    image_pil = Image.fromarray(image_rgba)

    # Save the image in Pillow format
    image_pil.save(os.path.join(saving_path, 'saved_gray_pil.png'))


def convert_pillow_to_opencv(photo_path, saving_path):
    image_pil = Image.open(photo_path)

    # Convert to OpenCV format (RGBA to BGRA)
    image_rgba = np.array(image_pil)
    if image_rgba.shape[2] == 4:
        image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
    else:
        image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGB2BGR)

    # Save the image in OpenCV format
    cv2.imwrite(os.path.join(saving_path, 'converted_back_image.jpg'), image_bgra)


if __name__ == '__main__':
    # your path to photo (or use this example)
    # Im using os.path.join to correctly connect path for Windows and MACos
    path_to_photo = os.path.join("seminars", "first_seminar_14.01.25", "image.jpg") 
    
    saving_path = os.path.join('seminars', 'first_seminar_14.01.25', 'created_data')
    if not os.path.exists(saving_path): # creating folder for saving photos
        os.makedirs(saving_path)

    """
    Uncomment tasks to check what they do
    """

    # # 2nd task
    # load_display_OpenCV(path_to_photo)
    # load_display_Pillow(path_to_photo)

    # # 3rd task
    # save_photo_OpenCV(path_to_photo, saving_path)
    # save_photo_Pillow(path_to_photo, saving_path)

    # # 4th task
    # image_chennels_OpenCV(path_to_photo, saving_path)
    # image_channels_pillow(path_to_photo, saving_path)

    # # 5th task 
    # resize_and_crop_opencv(path_to_photo, saving_path)
    # resize_and_crop_pillow(path_to_photo, saving_path)

    # # 6th task 
    # rotate_image_opencv(path_to_photo, saving_path)
    # rotate_image_pillow(path_to_photo, saving_path)

    # # 7th task
    # adjust_contrast_brightness_opencv(path_to_photo, saving_path)
    # adjust_contrast_brightness_pillow(path_to_photo, saving_path)

    # # 8th task
    # histogram_opencv(path_to_photo, saving_path)
    # histogram_pillow(path_to_photo, saving_path)

    # # 10th task
    # convert_opencv_to_pillow(path_to_photo, saving_path)
    # convert_pillow_to_opencv(path_to_photo, saving_path)