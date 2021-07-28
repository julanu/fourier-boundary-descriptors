import numpy as np
import cv2
from matplotlib import pyplot as plt


def convert_to_bw(filename):
    """
    Function to convert a given image to black and white based on pixel colors.  Bitonal.

    :param filename - relative file path for the iamge to be processed:
    :return img - B&W processed image:
    """

    # Read as grayscale
    img = cv2.imread(filename, 0)

    # Compute rows and columns length
    col = img.shape[1]
    row = img.shape[0]

    # Iterate over map
    for i in range(0, row):
        for j in range(0, col):
            
            # Check pixels color attributes and based on the values [x,y,z] decide to set it
            # to black or white
            pixel = img[i][j]
            # print(pixel)
            if (pixel > 230).all():
                img[i][j] = 255
            else:
                img[i][j] = 0

    return img


def apply_fft(img):
    """
    Function to compute the Discrete Fourier Transform of the image. Converting from spatial domain
    to frequency domain.

    :param img - binary image to be processed:
    :return magnitude_spectrum - spectrum of the image
            dft_shift - transform of the image:
    """

    # Apply discrete fourier transform
    # Flags: DFT_COMPLEX OUTPUT
    #        DFT_REAL_OUTPUT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift all the zero frequency components to the center of the spectrum
    # and compute the magnitude for the frequencies array
    dft_shift = np.fft.fftshift(dft)
    # print(dft_shift)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # print(magnitude_spectrum)

    return magnitude_spectrum, dft_shift


def gen_mask(img, dft_shift):
    """
    Function to create the mask for filtering the frequencies. The mask is created the same size as 
    the array of the image, having a circle of zerios in the center and the rest all ons. Applied 
    on the original image the resultant image will only have the high frequencies.

    :param img - binary image to extract properties and process:
    :param dshift - Discrete Fourier Transform of the spectrum with zeros shifted to the center:
    :return fshift - spectrum with mask applied for filtering the high frequencies
            fshift_mask_mag - frequencies spectrum after applying the mask and increasing the magnitude for plotting:
    """
    # Circular HPF mask, center circle is 0, remaining all ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center

    # Create an array for masking
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    r = 100

    # Open mesh-grid when indexed using ogrid, so that only one dimension
    # of each returned array is greater than 1
    x, y = np.ogrid[:rows, :cols]
    
    # Make around center of the mask zeros 
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    # print(mask)

    fshift = dft_shift * mask
    # print(fshift)

    # Compute the magnite spectrum after applying the Fourier Transform and the mask on the freqs array
    fshift_mask_mag = 12000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    return fshift, fshift_mask_mag


def apply_fft_inverse(fshift):
    """
    Function to apply the mask on the frequencies spectrum and the Inverse Fourier transform
    to obtain the edges of the input binary image
    :param img - image to be processed:
    :param fshift - array same size as the image with 1 & 0's to be applied on the spectrum:
    :return img_edge - binary image containing the edges using the magnitude() function:
    """

    # Apply the inverse Fourier transform shifting of the high frequencies 
    # to the center of the spectrum
    f_ishift = np.fft.ifftshift(fshift)
    
    # Apply Inverse Discrete Fourier Transform
    img_edge = cv2.idft(f_ishift)

    # Use the magnitude function to reconstruct the image from the points arrays
    # of the leaf margins
    img_edge = cv2.magnitude(img_edge[:, :, 0], img_edge[:, :, 1])

    return img_edge


def main():

    filename = ".\\imgs\\oak_leaf.jpg"
    

    img = cv2.imread(filename)
    plt.subplot(2, 3, 1), plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])


    bw_img = convert_to_bw(filename)
    plt.subplot(2, 3, 2), plt.imshow(bw_img, cmap='gray')
    plt.title('BW Input Image'), plt.xticks([]), plt.yticks([])
    

    spectrum, dft_shift = apply_fft(bw_img)
    plt.subplot(2, 3, 3), plt.imshow(spectrum, cmap='gray')
    plt.title('After FFT'), plt.xticks([]), plt.yticks([])


    fshift, fshift_mask_mag = gen_mask(bw_img, dft_shift)
    plt.subplot(2, 3, 4), plt.imshow(fshift_mask_mag, cmap='gray')
    plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])

    img_edge = apply_fft_inverse(fshift)
    plt.subplot(2, 3, 5), plt.imshow(img_edge, cmap='gray')
    plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
    
    plt.show()


    # FILTERING


if __name__ == '__main__':
    main()