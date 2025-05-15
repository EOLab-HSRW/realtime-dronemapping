import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class OrthomosaicRT:
    """Real-time orthomosaic generator that processes images as they arrive."""
    
    def __init__(self, image_folder, output_folder, update_interval=2.0):
        """
        Initialize the real-time orthomosaic generator.
        
        Args:
            image_folder: Folder to monitor for new drone images
            output_folder: Folder to save outputs
            update_interval: How often to update the orthomosaic (seconds)
        """
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.update_interval = update_interval
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Store images, keypoints, descriptors and transformations
        self.images = {}  # {filename: image}
        self.keypoints = {}  # {filename: keypoints}
        self.descriptors = {}  # {filename: descriptors}
        self.image_positions = {}  # {filename: homography matrix}
        
        # Feature detection and matching
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Canvas for the mosaic
        self.canvas = None
        self.canvas_shape = None
        
        # Queue for new images to process
        self.image_queue = queue.Queue()
        
        # Thread for processing images
        self.processing_thread = None
        self.is_running = False
        
        # Reference image (first image will be the reference)
        self.reference_img_name = None
        
        # For visualization
        self.visualize = True
        self.viz_thread = None
    
    def start_monitoring(self):
        """Start monitoring the image folder for new images."""
        self.is_running = True
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # If visualization is enabled, start visualization thread
        if self.visualize:
            self.viz_thread = threading.Thread(target=self._visualization_loop)
            self.viz_thread.daemon = True
            self.viz_thread.start()
        
        # Load any existing images in the folder
        self._load_existing_images()
        
        # Set up folder monitoring
        event_handler = ImageHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.image_folder), recursive=False)
        observer.start()
        
        print(f"Monitoring {self.image_folder} for new images...")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            observer.stop()
        
        observer.join()
    
    def stop(self):
        """Stop all processing threads."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        if self.viz_thread:
            self.viz_thread.join(timeout=2.0)
        print("Stopped orthomosaic generation")
    
    def _load_existing_images(self):
        """Load any existing images in the monitored folder."""
        image_files = sorted([f for f in self.image_folder.glob('*.jpg') or self.image_folder.glob('*.tif')])
        for img_path in image_files:
            self.queue_image(img_path)
    
    def queue_image(self, image_path):
        """Add a new image to the processing queue."""
        self.image_queue.put(image_path)
        print(f"Queued image: {image_path.name}")
    
    def _process_queue(self):
        """Process images from the queue to update the orthomosaic."""
        while self.is_running:
            try:
                # Get the next image path from the queue (wait up to 1 second)
                img_path = self.image_queue.get(timeout=1.0)
                
                # Process this image
                self._process_image(img_path)
                
                # Update the mosaic
                self._update_orthomosaic()
                
                # Save the current state
                self._save_current_mosaic()
                
                # Mark this task as done
                self.image_queue.task_done()
                
            except queue.Empty:
                # No new images, just sleep for a bit
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    
    def _process_image(self, img_path):
        """
        Process a new image:
        1. Load and preprocess
        2. Extract features
        3. Match with existing images
        4. Update transformations
        """
        img_name = img_path.name
        
        # Check if we've already processed this image
        if img_name in self.images:
            return
        
        # Load the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return
        
        # Convert to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store the image
        self.images[img_name] = img
        
        # Extract features
        kp, des = self.feature_detector.detectAndCompute(img, None)
        self.keypoints[img_name] = kp
        self.descriptors[img_name] = des
        
        print(f"Processed {img_name}: {len(kp)} keypoints")
        
        # If this is the first image, set it as reference
        if self.reference_img_name is None:
            self.reference_img_name = img_name
            self.image_positions[img_name] = np.eye(3)  # Identity transformation
        else:
            # Match with existing images and update transformations
            self._match_and_transform(img_name)
    
    def _match_and_transform(self, img_name, min_matches=10, ratio_threshold=0.75):
        """Match the new image with existing ones and update transformations."""
        best_matches = 0
        best_match_name = None
        best_homography = None
        
        # Current image keypoints and descriptors
        kp_current = self.keypoints[img_name]
        des_current = self.descriptors[img_name]
        
        # Try to match with each existing transformed image
        for ref_name in self.image_positions.keys():
            if ref_name == img_name:
                continue
            
            kp_ref = self.keypoints[ref_name]
            des_ref = self.descriptors[ref_name]
            
            # Match descriptors (k-nearest neighbor with k=2)
            raw_matches = self.matcher.knnMatch(des_current, des_ref, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for m, n in raw_matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            
            # Check if we have enough matches
            if len(good_matches) >= min_matches:
                # If this is the best match so far, compute homography
                if len(good_matches) > best_matches:
                    # Extract matched keypoints
                    src_pts = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Compute homography
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        inliers = np.sum(mask)
                        print(f"Match {img_name} to {ref_name}: {inliers} inliers out of {len(good_matches)}")
                        
                        if inliers > best_matches:
                            best_matches = inliers
                            best_match_name = ref_name
                            best_homography = H
        
        # If we found a good match, update the transformation
        if best_match_name is not None:
            # Compute the transformation from reference to current image
            # H_ref_to_current = H_ref_to_best * inv(H_current_to_best)
            ref_to_best = self.image_positions[best_match_name]
            current_to_best = best_homography
            
            # The transformation from reference to current image
            try:
                ref_to_current = ref_to_best @ np.linalg.inv(current_to_best)
                self.image_positions[img_name] = ref_to_current
                print(f"Added transformation for {img_name} (matches with {best_match_name})")
                return True
            except np.linalg.LinAlgError:
                print(f"Warning: Could not compute transformation for {img_name}")
                return False
        else:
            print(f"Warning: No good matches found for {img_name}")
            return False
    
    def _update_orthomosaic(self):
        """Update the orthomosaic with all transformed images."""
        if not self.image_positions:
            return
        
        # If first time or significant change, recompute canvas size
        if self.canvas is None or len(self.image_positions) % 5 == 0:
            self._create_canvas()
        
        # Clear canvas (create a fresh one each time)
        self.canvas.fill(0)
        
        # For each image with a transformation, warp it onto the canvas
        for img_name, H in self.image_positions.items():
            img = self.images[img_name]
            
            # Warp the image
            warped = cv2.warpPerspective(img, H, (self.canvas_shape[1], self.canvas_shape[0]))
            
            # Create a mask for the warped image
            mask = np.all(warped > 0, axis=2).astype(np.uint8)
            
            # Blend with existing canvas
            # Simple approach: take maximum value
            self.canvas = np.maximum(self.canvas, warped)
    
    def _create_canvas(self):
        """Create or resize canvas to fit all transformed images."""
        # Find the extremes of the transformed image corners
        corners_x = []
        corners_y = []
        
        for img_name, H in self.image_positions.items():
            img = self.images[img_name]
            h, w = img.shape[:2]
            
            # Get the four corners of the image
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1],
                [w, h, 1],
                [0, h, 1]
            ])
            
            # Transform the corners
            transformed_corners = H @ corners.T
            
            # Normalize homogeneous coordinates
            transformed_corners = transformed_corners / transformed_corners[2, :]
            
            corners_x.extend(transformed_corners[0, :])
            corners_y.extend(transformed_corners[1, :])
        
        # Calculate canvas dimensions
        min_x, max_x = min(corners_x), max(corners_x)
        min_y, max_y = min(corners_y), max(corners_y)
        
        # Add padding
        width = int(max_x - min_x) + 200
        height = int(max_y - min_y) + 200
        
        # Create translation to ensure all points are within canvas
        translation = np.array([
            [1, 0, -min_x + 100],
            [0, 1, -min_y + 100],
            [0, 0, 1]
        ])
        
        # Update all transformations with this translation
        for img_name in self.image_positions:
            self.image_positions[img_name] = translation @ self.image_positions[img_name]
        
        # Create empty canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas_shape = (height, width)
        
        print(f"Canvas resized to {width}x{height}")
    
    def _save_current_mosaic(self):
        """Save the current orthomosaic."""
        if self.canvas is None:
            return
            
        timestamp = int(time.time())
        output_path = self.output_folder / f"orthomosaic_{timestamp}.jpg"
        
        # Convert from RGB to BGR for OpenCV's imwrite
        output_img = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output_img)
        
        # Also save a copy as "latest.jpg"
        latest_path = self.output_folder / "orthomosaic_latest.jpg"
        cv2.imwrite(str(latest_path), output_img)
        
        print(f"Saved orthomosaic to {output_path}")

    def _visualization_loop(self):
        """Thread for showing the orthomosaic in real-time."""
        plt.figure(figsize=(12, 10))
        plt.ion()  # Interactive mode
        
        img_obj = None
        
        while self.is_running:
            if self.canvas is not None:
                if img_obj is None:
                    img_obj = plt.imshow(self.canvas)
                    plt.title("Real-time Orthomosaic")
                    plt.axis('off')
                else:
                    img_obj.set_data(self.canvas)
                    plt.draw()
                
                plt.pause(0.5)
            else:
                time.sleep(0.5)


class ImageHandler(FileSystemEventHandler):
    """Handler for watching the image folder for new files."""
    
    def __init__(self, orthomosaic_generator):
        self.generator = orthomosaic_generator
    
    def on_created(self, event):
        """Called when a file is created in the watched folder."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            # Check if it's an image file
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', 'tif', 'tiff']:
                # Add to the processing queue
                self.generator.queue_image(file_path)


# Usage example
if __name__ == "__main__":
    # Folders for input images and output mosaics
    image_folder = "/home/azam/images/"
    output_folder = "output/"
    
    # Create and start the real-time orthomosaic generator
    mosaic_generator = OrthomosaicRT(image_folder, output_folder)
    mosaic_generator.start_monitoring()
    
    # The generator will run until Ctrl+C is pressed
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        mosaic_generator.stop()