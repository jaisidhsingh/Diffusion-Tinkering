import cv2
import numpy as np
import pickle


# Global variables
image_name = "lsun-bedroom-scenes_1_4.png"

f = open("./masks/info.pkl", "rb")
data = pickle.load(f)

image = None
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, data

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(mask, (ix, iy), (x, y), (255), thickness=cv2.FILLED)
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        
        print((ix, iy), (x, y))
        cv2.imwrite(f'./masks/{image_name}', image)
        data[image_name] = [ix, iy, x, y]
        with open("./masks/info.pkl", "wb") as f:
            pickle.dump(data, f)

def apply_mask(image, mask):
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def main():
    global image, mask

    # Load the image
    image = cv2.imread(f'../image_data/lsun-bedroom-scenes/{image_name}')
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create a window to display the image
    cv2.namedWindow('Masked Image', cv2.WINDOW_NORMAL)

    # Set the callback function for mouse events
    cv2.setMouseCallback('Masked Image', draw_rectangle)

    while True:
        cv2.imshow('Masked Image', image)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Apply the mask to the image
    masked_image = apply_mask(image, mask)

    # Display the masked image
    cv2.imshow('Masked Image', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
