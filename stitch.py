# This file contains all the stitching code I have tested so far, apart from the existing code that I have tested. 

'''
import cv2
import numpy as np
import os
from glob import glob
from stitching import AffineStitcher

IMAGE_FOLDER = "images/"

def load_images(image_folder):
    """Loads and sorts images from a folder."""
    image_paths = sorted(glob(os.path.join(image_folder, "*.JPG")))  # Adjust for other formats if needed
    images = [cv2.imread(img) for img in image_paths]
    return images

def detect_features(image, scale_factor=0.5):
    """Detects keypoints and descriptors using ORB on a resized image."""
    h, w = image.shape[:2]
    small_image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(small_image, None)
    # Scale keypoints back to original image size
    for kp in keypoints:
        kp.pt = (kp.pt[0] / scale_factor, kp.pt[1] / scale_factor)
    return keypoints, descriptors

def match_features(des1, des2):
    """Matches keypoints using KNN and Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def compute_homography(kp1, kp2, matches):
    """Computes the homography matrix using RANSAC."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def crop_black_borders(image):
    """Crops black borders from the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y:y+h, x:x+w]
    return cropped

def stitch_images(images):
    """Stitches images together efficiently."""
    base_image = images[0]
    
    for i in range(1, len(images)):
        print(f"Stitching image {i+1} of {len(images)}...")
        # Detect features with resizing
        kp1, des1 = detect_features(base_image)
        kp2, des2 = detect_features(images[i])
        
        # Match features
        matches = match_features(des1, des2)
        if len(matches) < 4:
            print(f"Skipping image {i+1} due to insufficient matches.")
            continue
        
        # Compute homography and invert it
        H, mask = compute_homography(kp2, kp1, matches)  # Note kp2 and kp1 swapped for inverse
        H_inv = np.linalg.inv(H)
        
        # Calculate canvas size
        h_new, w_new = images[i].shape[:2]
        h_base, w_base = base_image.shape[:2]
        corners = np.array([[0, 0], [w_new, 0], [w_new, h_new], [0, h_new]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H_inv).reshape(-1, 2)
        base_corners = np.array([[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]], dtype=np.float32)
        all_corners = np.concatenate((warped_corners, base_corners), axis=0)
        
        # Determine canvas bounds
        min_x = int(np.floor(all_corners[:, 0].min()))
        min_y = int(np.floor(all_corners[:, 1].min()))
        max_x = int(np.ceil(all_corners[:, 0].max()))
        max_y = int(np.ceil(all_corners[:, 1].max()))
        new_width = max_x - min_x
        new_height = max_y - min_y
        
        # Adjust homography for translation
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_adjusted = translation.dot(H_inv)
        
        # Warp new image
        warped_image = cv2.warpPerspective(images[i], H_adjusted, (new_width, new_height))
        
        # Warp base image
        base_translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        warped_base = cv2.warpPerspective(base_image, base_translation, (new_width, new_height))
        
        # Combine images
        mask = warped_image.sum(axis=2) > 0
        warped_base[mask] = warped_image[mask]
        stitched = warped_base
        
        # Crop and update base image
        stitched = crop_black_borders(stitched)
        base_image = stitched
    
    return base_image

def save_and_show_mosaic(mosaic):
    cv2.imwrite("orthomosaic.jpg", mosaic)
    #cv2.imshow("Orthomosaic", mosaic)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def main():
    images = load_images(IMAGE_FOLDER)
    if len(images) < 2:
        print("Need at least 2 images.")
        return
    
    #stitcher = AffineStitcher(detector="brisk", crop=False)
    #mosaic = stitcher.stitch(images)
    #mosaic = stitch_images(images)
    #save_and_show_mosaic(mosaic)
   
if __name__ == "__main__":
    main()
''' 
import os
import sys
sys.path.append('..')

from pyodm import Node, exceptions

node = Node("localhost", 3000)

try:
    # Start a task
    print("Uploading images...")
    task = node.create_task(['images/DJI_0812.JPG', 
                             'images/DJI_0813.JPG'
                             ],
                            {'orthophoto-resolution': 1, 
                             'fast-orthophoto': True, 
                             'orthophoto-png': True, 
                             'skip-report': True, 
                             'skip-3dmodel': True, 
                             'auto_boundary': True,
                             'orthophoto-cutline': True,
                             'auto-boundary': True})
    print(task.info())

    try:
        # This will block until the task is finished
        # or will raise an exception
        task.wait_for_completion()

        print("Task completed, downloading results...")

        # Retrieve results
        task.download_assets("./results")

        print("Assets saved in ./results (%s)" % os.listdir("./results"))

        # Restart task and this time compute dtm
        #task.restart({'dtm': True})
        #task.wait_for_completion()

        #print("Task completed, downloading results...")

        #task.download_assets("./results_with_dtm")

        #print("Assets saved in ./results_with_dtm (%s)" % os.listdir("./results_with_dtm"))
    except exceptions.TaskFailedError as e:
        print("\n".join(task.output()))

except exceptions.NodeConnectionError as e:
    print("Cannot connect: %s" % e)
except exceptions.NodeResponseError as e:
    print("Error: %s" % e)
