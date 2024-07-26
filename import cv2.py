import cv2
import os

def collect_images(person_name, num_images=500):
    # Create a directory for the person if it doesn't exist
    save_dir = rf'C:\Users\User\Desktop\rec project\dataset\{person_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Image Collection', frame)

        # Save the frame to the dataset
        img_path = os.path.join(save_dir, f'{person_name}_{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
collect_images('omar emad', 50)
collect_images('omar emad 2', 50)