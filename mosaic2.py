import os
import cv2
import numpy as np
from pathlib import Path
import gc  # Garbage collection for memory management
import math
import datetime
import time

def extract_gps_from_exif(image_path):
    """
    Extract GPS coordinates from image EXIF data if available.
    Returns (latitude, longitude, altitude) or None if not available.
    """
    try:
        # Using OpenCV to read EXIF data
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Try to extract EXIF data using OpenCV's EXIF reading capability
        exif_data = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        # Real implementation would extract actual GPS data
        # This is a placeholder since OpenCV doesn't directly support EXIF GPS reading
        # For actual implementation, you would use PIL or exifread libraries
        return None
    except:
        return None

def find_matching_features(img1, img2, feature_method='sift', match_ratio=0.75, max_features=500):
    """
    Find matching features between two images using SIFT, SURF, or ORB
    
    Parameters:
    - img1, img2: The two images to match
    - feature_method: 'sift', 'orb', or 'akaze'
    - match_ratio: Ratio test threshold for feature matching
    - max_features: Maximum number of features to detect (memory control)
    
    Returns:
    - List of good matches, keypoints for img1 and img2
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector with controlled number of features
    if feature_method == 'sift':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif feature_method == 'orb':
        detector = cv2.ORB_create(nfeatures=max_features)
    elif feature_method == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.SIFT_create(nfeatures=max_features)  # Default to SIFT
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    
    if descriptors1 is None or descriptors2 is None or len(descriptors1) < 2 or len(descriptors2) < 2:
        return [], [], []
    
    # Feature matching
    if feature_method == 'orb' or feature_method == 'akaze':
        # Use Hamming distance for binary descriptors (ORB, AKAZE)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # Use L2 norm for SIFT
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Compute matches
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance:
            good_matches.append(m)
    
    return good_matches, keypoints1, keypoints2

def estimate_drone_path(image_paths, max_image_dim=800):
    """
    Estimate the path the drone took based on feature matching between consecutive images.
    Uses a much lower resolution for this estimation to save memory.
    
    Parameters:
    - image_paths: List of paths to drone images
    - max_image_dim: Maximum dimension for downsized images during path estimation
    
    Returns:
    - Ordered list of images based on the estimated path
    """
    print("Estimating drone flight path from image sequence...")
    
    if len(image_paths) <= 2:
        # If only 1-2 images, no need to reorder
        return image_paths
    
    # Try to read EXIF data first for drone path estimation
    gps_data = {}
    for path in image_paths:
        gps = extract_gps_from_exif(path)
        if gps:
            gps_data[path] = gps
    
    # If we have GPS data for most images, use that to order images
    if len(gps_data) >= len(image_paths) * 0.7:  # At least 70% have GPS
        print("Using GPS data to order images")
        # Sort by GPS coordinates (this is simplified - actual implementation 
        # would create a proper path through GPS coordinates)
        return sorted(image_paths)
    
    # Otherwise, use feature matching to determine image sequence
    print("GPS data not available. Using image features to determine sequence...")
    
    # Start with the first image
    sequence = [image_paths[0]]
    remaining = set(image_paths[1:])
    
    # Very aggressive downscale factor for path estimation only
    scale_factor = max_image_dim / 3000  # Assuming most drone images are around 3000px
    
    # Process the first image
    last_img_path = sequence[-1]
    last_img = cv2.imread(str(last_img_path))
    if last_img is None:
        # If we can't read the first image, just return original order
        print("Warning: Could not read first image for path estimation")
        return image_paths
        
    h, w = last_img.shape[:2]
    # Ensure we use a very small image for path estimation
    target_size = (min(int(w*scale_factor), max_image_dim), min(int(h*scale_factor), max_image_dim))
    last_img = cv2.resize(last_img, target_size)
    
    while remaining:
        best_match_count = 0
        next_img_path = None
        
        # Process in batches to avoid memory issues
        batch_size = min(10, len(remaining))
        batch_paths = list(remaining)[:batch_size]
        
        for path in batch_paths:
            # Read and resize current candidate
            curr_img = cv2.imread(str(path))
            if curr_img is None:
                remaining.remove(path)
                continue
                
            h, w = curr_img.shape[:2]
            target_size = (min(int(w*scale_factor), max_image_dim), min(int(h*scale_factor), max_image_dim))
            curr_img = cv2.resize(curr_img, target_size)
            
            # Find matches between last image and current candidate
            # Use lower max_features for path estimation
            matches, _, _ = find_matching_features(last_img, curr_img, max_features=300)
            match_count = len(matches)
            
            # Update best match
            if match_count > best_match_count:
                best_match_count = match_count
                next_img_path = path
                
            # Free memory
            del curr_img
            gc.collect()
        
        if next_img_path:
            # Add the best match to sequence and remove from remaining
            sequence.append(next_img_path)
            remaining.remove(next_img_path)
            
            # Update last image
            last_img_path = next_img_path
            last_img = cv2.imread(str(last_img_path))
            if last_img is not None:
                h, w = last_img.shape[:2]
                target_size = (min(int(w*scale_factor), max_image_dim), min(int(h*scale_factor), max_image_dim))
                last_img = cv2.resize(last_img, target_size)
        else:
            # If no good match found, just add the first remaining image
            next_path = next(iter(remaining))
            sequence.append(next_path)
            remaining.remove(next_path)
            
            # Update last image
            last_img_path = next_path
            last_img = cv2.imread(str(last_img_path))
            if last_img is not None:
                h, w = last_img.shape[:2]
                target_size = (min(int(w*scale_factor), max_image_dim), min(int(h*scale_factor), max_image_dim))
                last_img = cv2.resize(last_img, target_size)
        
        print(f"Processed {len(sequence)}/{len(image_paths)} images in sequence")
        # Force garbage collection
        gc.collect()
    
    print(f"Determined image sequence order. First image: {sequence[0].name}")
    return sequence

def stitch_drone_images(input_folder, output_path, max_input_dim=1000, max_output_dim=4000, 
                       feature_method='sift', use_drone_path=True, warp_mode='plane', 
                       memory_limit_gb=4, confidence=0.8):
    """
    Specialized function for stitching drone images with memory optimization
    
    Parameters:
    - input_folder: Path to folder containing drone images
    - output_path: Path where the stitched image will be saved
    - max_input_dim: Maximum dimension for input images before stitching
    - max_output_dim: Maximum dimension for output panorama
    - feature_method: Feature detection method ('sift', 'orb', 'akaze')
    - use_drone_path: Whether to estimate and use the drone path for ordering images
    - warp_mode: Type of warping for stitching ('spherical', 'plane', 'cylindrical')
    - memory_limit_gb: Approximate memory limit in GB to stay within
    - confidence: Confidence for stitcher (0.0-1.0)
    
    Returns:
    - Path to stitched image if successful, None otherwise
    """
    print(f"Starting drone image stitching process for {input_folder}")
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_paths.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))
    
    if not image_paths:
        print("No images found in the specified folder.")
        return None
    
    # Sort images - first by default alphabetical order
    image_paths.sort()
    
    # If requested, try to determine the drone path for better image ordering
    if use_drone_path and len(image_paths) > 2:
        # Use a smaller max dimension for path estimation to save memory
        image_paths = estimate_drone_path(image_paths, max_image_dim=400)
    
    # Memory management logic
    # Estimate memory requirement and adjust input dimensions if needed
    sample_img = cv2.imread(str(image_paths[0]))
    if sample_img is None:
        print("Could not read sample image for memory estimation")
        return None
        
    # Calculate approx memory per image at max_input_dim
    h, w = sample_img.shape[:2]
    scale = max_input_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    bytes_per_image = new_h * new_w * 4 * 3  # Approx bytes per image (BGR + working memory)
    
    # Total images to process
    num_images = len(image_paths)
    
    # Estimate total memory requirement
    total_memory_bytes = bytes_per_image * num_images * 2  # Extra factor for stitching overhead
    total_memory_gb = total_memory_bytes / (1024**3)
    
    # If estimated memory exceeds limit, reduce dimensions
    if total_memory_gb > memory_limit_gb:
        # Calculate adjustment factor to fit in memory
        adjustment_factor = math.sqrt(memory_limit_gb / total_memory_gb)
        max_input_dim = int(max_input_dim * adjustment_factor)
        print(f"Memory optimization: Adjusted max input dimension to {max_input_dim} to fit in {memory_limit_gb}GB memory")
    
    print(f"Processing {len(image_paths)} images with max dimension {max_input_dim}px...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process images in batches if there are many
    batch_size = min(5, num_images)  # Start with 5 images per batch
    if num_images > 20:  # If we have many images, use smaller batches
        batch_size = 3
    
    # Read and resize images to control memory usage
    print(f"Reading and resizing images in batches of {batch_size}...")
    
    # Split images into manageable batches
    batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    batch_results = []
    
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)}...")
        
        batch_images = []
        for path in batch:
            try:
                # Read the image
                img = cv2.imread(str(path))
                if img is None:
                    print(f"Warning: Could not read image {path}")
                    continue
                    
                # Get dimensions
                h, w = img.shape[:2]
                
                # Resize if larger than max_input_dim
                if max(h, w) > max_input_dim:
                    scale = max_input_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"Resized {path.name} from {w}x{h} to {new_w}x{new_h}")
                    batch_images.append(resized_img)
                    del img  # Free memory
                else:
                    batch_images.append(img)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        # Skip if there are not enough images in this batch
        if len(batch_images) < 2:
            print(f"Batch {batch_idx+1} has less than 2 valid images, skipping...")
            continue
        
        # Process this batch
        try:
            # Configure stitcher
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            
            # Try to set confidence parameter if available
            try:
                stitcher.setConfidenceThresh(confidence)
            except:
                print("Could not set confidence threshold - using default")
            
            # Try to set feature finder based on our method
            # Note: We can't directly set these in Python API, but we can adjust other params
            if feature_method == 'orb':
                try:
                    stitcher.setFeaturesFinder(cv2.ORB_create(nfeatures=1500))
                except:
                    print("Could not set ORB feature finder")
            
            print(f"Stitching batch {batch_idx+1}...")
            status, result = stitcher.stitch(batch_images)
            
            if status == cv2.Stitcher_OK:
                batch_results.append(result)
                print(f"Successfully stitched batch {batch_idx+1}")
            else:
                print(f"Failed to stitch batch {batch_idx+1}, error code: {status}")
                
                # Try again with fewer images if batch size > 2
                if len(batch_images) > 2:
                    print("Trying to stitch batch with smaller groups...")
                    
                    # Split into smaller sub-batches
                    sub_batches = [batch_images[i:i+2] for i in range(0, len(batch_images)-1)]
                    
                    for sub_idx, sub_batch in enumerate(sub_batches):
                        if len(sub_batch) >= 2:
                            sub_stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                            sub_status, sub_result = sub_stitcher.stitch(sub_batch)
                            
                            if sub_status == cv2.Stitcher_OK:
                                batch_results.append(sub_result)
                                print(f"Successfully stitched sub-batch {sub_idx+1}")
            
        except Exception as e:
            print(f"Error during batch stitching: {e}")
        
        # Clean up to free memory
        batch_images = None
        gc.collect()
    
    # If we have batch results, stitch them together
    if len(batch_results) >= 2:
        print(f"Stitching {len(batch_results)} batch results together...")
        try:
            final_stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, result = final_stitcher.stitch(batch_results)
            
            if status != cv2.Stitcher_OK:
                print(f"Failed to stitch batches together, error code: {status}")
                
                # If we can't stitch all batches together, return the largest batch result
                if batch_results:
                    largest_idx = max(range(len(batch_results)), 
                                     key=lambda i: batch_results[i].shape[0] * batch_results[i].shape[1])
                    result = batch_results[largest_idx]
                    print("Using largest successful batch as final result")
                else:
                    return None
        except Exception as e:
            print(f"Error during final stitching: {e}")
            
            # Use the largest batch result
            if batch_results:
                largest_idx = max(range(len(batch_results)), 
                                 key=lambda i: batch_results[i].shape[0] * batch_results[i].shape[1])
                result = batch_results[largest_idx]
                print("Using largest successful batch as final result due to error")
            else:
                return None
    elif len(batch_results) == 1:
        # If only one batch result, use it directly
        result = batch_results[0]
    else:
        print("No successful batch stitching results")
        return None
    
    print("Stitching successful! Processing final image...")
    
    # Free memory
    batch_results = None
    gc.collect()
    
    # Resize output if too large
    h, w = result.shape[:2]
    if max(h, w) > max_output_dim:
        scale_factor = max_output_dim / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized final image from {w}x{h} to {new_w}x{new_h}")
    
    # Crop black borders
    result = crop_black_borders(result)
    
    # Save the result with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.splitext(output_path)[0]
    ext = os.path.splitext(output_path)[1]
    final_path = f"{base_path}_{timestamp}{ext}"
    
    compression_level = 90  # Higher quality for drone imagery
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, compression_level]
    cv2.imwrite(final_path, result, encode_params)
    
    print(f"Stitched image saved to: {final_path}")
    return final_path

def crop_black_borders(img):
    """
    Crop black borders that often appear in stitched panoramas
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to find non-black areas
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours of non-black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Find the largest contour (the main panorama)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small buffer (1% of image size) to avoid cutting too tight
    buffer_x = int(w * 0.01)
    buffer_y = int(h * 0.01)
    
    # Make sure buffers don't go out of bounds
    x = max(0, x - buffer_x)
    y = max(0, y - buffer_y)
    w = min(img.shape[1] - x, w + 2 * buffer_x)
    h = min(img.shape[0] - y, h + 2 * buffer_y)
    
    # Crop the image to this rectangle
    cropped = img[y:y+h, x:x+w]
    return cropped

def enhance_drone_image(img_path, output_path, sharpen=True, contrast=True, color_balance=True):
    """
    Enhance a drone mosaic image with specialized processing
    
    Parameters:
    - img_path: Path to the input image
    - output_path: Path to save the enhanced image
    - sharpen: Whether to apply sharpening
    - contrast: Whether to enhance contrast
    - color_balance: Whether to balance colors across the image
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not open image at {img_path}")
            return None
        
        # Apply sharpening (with gentle parameters for drone imagery)
        if sharpen:
            kernel = np.array([[-0.3, -0.3, -0.3],
                              [-0.3, 3.4, -0.3],
                              [-0.3, -0.3, -0.3]])
            img = cv2.filter2D(img, -1, kernel)
        
        # Enhance contrast
        if contrast:
            # Split image into LAB channels
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            merged = cv2.merge((cl, a, b))
            img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Apply color balancing for more consistent colors across the mosaic
        if color_balance:
            # Simple white balancing
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.average(img_lab[:, :, 1])
            avg_b = np.average(img_lab[:, :, 2])
            
            # Adjust by shifting a and b channels to be centered at 128
            img_lab[:, :, 1] = img_lab[:, :, 1] - ((avg_a - 128) * 0.7)
            img_lab[:, :, 2] = img_lab[:, :, 2] - ((avg_b - 128) * 0.7)
            
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        
        # Save with high quality compression
        compression_level = 95  # Higher quality for final output
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, compression_level]
        cv2.imwrite(output_path, img, encode_params)
        
        print(f"Enhanced drone mosaic saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error during image enhancement: {e}")
        return None

def main():
    print("\n===== Drone Image Mosaic Creator (Memory-Optimized) =====\n")
    
    # Get folder path from user
    input_folder = input("Enter the path to the folder containing drone images: ")
    
    # Set output paths
    output_dir = os.path.join(os.path.dirname(input_folder), "drone_mosaic_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base output paths
    base_name = os.path.basename(input_folder.rstrip('/\\'))
    stitched_path = os.path.join(output_dir, f"{base_name}_mosaic.jpg")
    enhanced_path = os.path.join(output_dir, f"{base_name}_mosaic_enhanced.jpg")
    
    # Ask for parameters with safer defaults
    print("\nStitching parameters (optimized for memory efficiency):")
    
    # Input size - default much lower than before
    try:
        max_input = input("Maximum input image dimension (recommended 800-1200, default 1000): ")
        max_input_dim = int(max_input) if max_input.strip() else 1000
    except ValueError:
        print("Invalid input. Using default value 1000.")
        max_input_dim = 1000
    
    # Output size - also lower default
    try:
        max_output = input("Maximum output mosaic dimension (recommended 2000-6000, default 4000): ")
        max_output_dim = int(max_output) if max_output.strip() else 4000
    except ValueError:
        print("Invalid input. Using default value 4000.")
        max_output_dim = 4000
    
    # Memory limit
    try:
        memory_limit = input("Memory limit in GB (recommended 2-8, default 4): ")
        memory_limit_gb = float(memory_limit) if memory_limit.strip() else 4.0
    except ValueError:
        print("Invalid input. Using default value 4 GB.")
        memory_limit_gb = 4.0
    
    # Feature detection method
    print("\nFeature detection method:")
    print("1. SIFT (better quality, more memory usage)")
    print("2. ORB (faster, less memory usage)")
    print("3. AKAZE (good balance between quality and speed)")
    feature_choice = input("Choose feature detection method (1-3), default 2 for memory efficiency: ").strip()
    
    if feature_choice == '1':
        feature_method = 'sift'
    elif feature_choice == '3':
        feature_method = 'akaze'
    else:
        feature_method = 'orb'  # Default for memory efficiency
    
    # Warping type
    print("\nWarping type for drone mosaic:")
    print("1. Plane (flat surface mapping, best for aerial/drone images)")
    print("2. Spherical (standard panorama warping)")
    print("3. Cylindrical (vertical straight lines preserved)")
    warp_choice = input("Choose warping type (1-3), default 1 for drone imagery: ").strip()
    
    if warp_choice == '2':
        warp_mode = 'spherical'
    elif warp_choice == '3':
        warp_mode = 'cylindrical'
    else:
        warp_mode = 'plane'  # Default for drone imagery
    
    # Ask about drone path estimation
    path_choice = input("\nEstimate drone flight path for better image ordering? (y/n), default y: ").strip().lower()
    use_drone_path = path_choice != 'n'
    
    # Confidence parameter
    try:
        confidence = input("Confidence threshold (0.0-1.0, higher=more confident matches but fewer features, default 0.8): ")
        confidence_val = float(confidence) if confidence.strip() else 0.8
        confidence_val = max(0.1, min(1.0, confidence_val))  # Ensure in range
    except ValueError:
        print("Invalid input. Using default value 0.8.")
        confidence_val = 0.8
    
    # Start timing
    start_time = time.time()
    print("\nStarting drone image mosaic creation with memory optimization...\n")
    
    # Force garbage collection before starting
    gc.collect()
    
    # Stitch images
    result_path = stitch_drone_images(
        input_folder, 
        stitched_path,
        max_input_dim=max_input_dim,
        max_output_dim=max_output_dim,
        feature_method=feature_method,
        use_drone_path=use_drone_path,
        warp_mode=warp_mode,
        memory_limit_gb=memory_limit_gb,
        confidence=confidence_val
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    if result_path:
        print(f"\nStitching completed in {minutes} minutes {seconds} seconds.")
        
        # Ask if user wants to enhance the image
        enhance = input("Do you want to enhance the mosaic quality? (yes/no): ").lower().strip()
        if enhance in ['yes', 'y']:
            print("Enhancing mosaic quality...")
            enhance_drone_image(result_path, enhanced_path)
            print(f"\nProcess complete. Final drone mosaic saved to: {enhanced_path}")
        else:
            print(f"\nProcess complete. Final drone mosaic saved to: {result_path}")
    else:
        print(f"\nMosaic creation failed after {minutes} minutes {seconds} seconds.")
        print("Try further reducing the image dimensions or using a different feature detection method.")

if __name__ == "__main__":
    main()