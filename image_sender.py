import os
import socket
import struct
import time
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    """
    Extract EXIF data from an image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Extracted EXIF data or None if no GPS data found
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        # Create a dictionary of EXIF tags
        exif_info = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_info[tag_name] = value
        
        return exif_info
    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {e}")
        return None

def parse_gps_coordinate(coord_str, ref):
    """
    Parse GPS coordinate from a string, handling various formats.
    
    Args:
        coord_str (str): GPS coordinate as a string
        ref (str): Reference direction (N/S or E/W)
    
    Returns:
        float: Decimal degree coordinate
    """
    try:
        # Replace '/' with ',' to standardize
        coord_str = coord_str.replace('/', ',')
        
        # Split the string and convert to float
        parts = [float(p.strip()) for p in coord_str[1:-1:].split(',')]
        
        # Convert to decimal degrees
        if len(parts) == 3:
            # Degrees, minutes, seconds format
            decimal_degrees = (
                parts[0] + 
                (parts[1] / 60) + 
                (parts[2] / 3600)
            )
        elif len(parts) == 1:
            # Already in decimal degrees
            decimal_degrees = parts[0]
        else:
            print(f"Unexpected coordinate format: {coord_str}")
            return None
        
        # Apply direction reference
        return decimal_degrees * (-1 if ref in ['S', 'W'] else 1)
    
    except (ValueError, TypeError) as e:
        print(f"Error parsing coordinate {coord_str}: {e}")
        return None

def get_gps_coordinates(exif_data):
    """
    Extract GPS coordinates from EXIF data.
    
    Args:
        exif_data (dict): EXIF data dictionary
    
    Returns:
        tuple: (latitude, longitude) or None if no GPS data found
    """
    if not exif_data:
        return None
    
    # Check if GPS info exists
    gps_info = {}
    for key in exif_data:
        if key == 'GPSInfo':
            # Parse GPS info
            for t in exif_data[key]:
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = exif_data[key][t]
    
    # Extract latitude and longitude
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        # Convert coordinates to strings if they're not already
        lat = str(gps_info['GPSLatitude'])
        lon = str(gps_info['GPSLongitude'])
        
        # Get references
        lat_ref = gps_info.get('GPSLatitudeRef', 'N')
        lon_ref = gps_info.get('GPSLongitudeRef', 'E')
        
        # Parse coordinates
        parsed_lat = parse_gps_coordinate(lat, lat_ref)
        parsed_lon = parse_gps_coordinate(lon, lon_ref)
        
        # Return if both parsed successfully
        if parsed_lat is not None and parsed_lon is not None:
            return parsed_lat, parsed_lon
    
    return None

def send_image(client_socket, image_path):
    """
    Send image and its GPS data over socket.
    
    Args:
        client_socket (socket): Connected socket
        image_path (str): Path to the image file
    """
    # Get EXIF and GPS data
    exif_data = get_exif_data(image_path)
    gps_coords = get_gps_coordinates(exif_data)
    
    # Open the image file
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    # Prepare GPS coordinates (send as string or None)
    if gps_coords:
        lat, lon = gps_coords
        gps_data = f"{lat},{lon}"
    else:
        # Optional: you might want to add a default location
        gps_data = f"51.49794822222222,6.549198"  # Default location if no GPS data
    
    # Send GPS data first
    client_socket.sendall(gps_data.encode('utf-8'))
    # Wait for acknowledgement
    client_socket.recv(1024)
    
    # Send image size
    client_socket.sendall(struct.pack("!I", len(image_data)))
    # Wait for acknowledgement
    client_socket.recv(1024)
    
    # Send image data
    client_socket.sendall(image_data)

def main():
    # Server connection details
    HOST = 'localhost'  # Server IP
    PORT = 65432        # Port to connect to
    
    # Folder containing images
    IMAGE_FOLDER = './images'
    
    # Ensure the image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Image folder {IMAGE_FOLDER} does not exist!")
        return
    
    # List all image files
    image_files = [
        os.path.join(IMAGE_FOLDER, f) 
        for f in os.listdir(IMAGE_FOLDER) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]
    
    # Sort images to ensure consistent order
    image_files.sort()
    
    # Connect to server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
        
        # Send images
        for image_path in image_files:
            print(f"Sending image: {image_path}")
            send_image(client_socket, image_path)
            
            # Optional: add a small delay between image sends
            time.sleep(0.5)
        
        # Send termination signal
        client_socket.sendall("END".encode('utf-8'))
    
    print("All images sent successfully!")

if __name__ == "__main__":
    main()