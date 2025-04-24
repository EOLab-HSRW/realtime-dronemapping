import os
import socket
import struct
import folium
from io import BytesIO
from PIL import Image

class ImageMap:
    def __init__(self, output_path='realtime_map.html'):
        """
        Initialize the map with a default center point.
        
        Args:
            output_path (str): Path to save the HTML map file
        """
        # Default to a global view
        self.map = folium.Map(location=[51.49794822222222, 6.549198], zoom_start = 24, max_zoom = 24)
        self.output_path = output_path
        self.image_count = 0
    
    def add_image_marker(self, lat, lon, image_data):
        """
        Add a marker with the image to the map.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            image_data (bytes): Image data
        """
        # Increment image count for unique filenames
        self.image_count += 1
        
        # Save the image
        image_folder = './received_images'
        os.makedirs(image_folder, exist_ok=True)
        image_path = os.path.join(image_folder, f'image_{self.image_count}.jpg')
        
        # Convert image bytes to file
        image = Image.open(BytesIO(image_data))
        image.save(image_path)
        
        # Resize image for thumbnail (optional)
        image.thumbnail((200, 200))
        thumbnail_path = os.path.join(image_folder, f'thumb_{self.image_count}.jpg')
        image.save(thumbnail_path)
        
        # Add marker with popup image
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f'<img src="{thumbnail_path}" width="200px"><br>Image {self.image_count}',
                max_width=250
            ),
            icon=folium.Icon(color='blue', icon='camera')
        ).add_to(self.map)
        
        # Save updated map
        self.map.save(self.output_path)
        print(f"Added image {self.image_count} at ({lat}, {lon})")

def receive_image(server_socket):
    """
    Receive image and GPS data from client.
    
    Args:
        server_socket (socket): Connected client socket
    
    Returns:
        tuple: (GPS coordinates, image data) or None if transmission ends
    """
    # Receive GPS data
    gps_data = server_socket.recv(1024).decode('utf-8')
    
    # Check for end of transmission
    if gps_data == "END":
        return None
    
    # Acknowledge GPS data receipt
    server_socket.sendall(b"GPS_RECEIVED")
    
    # Receive image size
    size_data = server_socket.recv(4)
    image_size = struct.unpack("!I", size_data)[0]
    
    # Acknowledge size receipt
    server_socket.sendall(b"SIZE_RECEIVED")
    
    # Receive image data
    image_data = b''
    while len(image_data) < image_size:
        chunk = server_socket.recv(min(4096, image_size - len(image_data)))
        if not chunk:
            break
        image_data += chunk
    
def parse_gps_coordinate(coord_str):
    """
    Parse GPS coordinate from a string, handling various formats.
    
    Args:
        coord_str (str): GPS coordinate as a string
    
    Returns:
        float: Parsed coordinate value
    """
    try:
        # Replace '/' with ',' to standardize
        coord_str = coord_str.replace('/', ',')
        
        # Split the string and convert to float
        parts = [float(p.strip()) for p in coord_str.split(',')]
        
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
        
        return decimal_degrees
    
    except (ValueError, TypeError) as e:
        print(f"Error parsing coordinate {coord_str}: {e}")
        return None

def receive_image(server_socket):
    """
    Receive image and GPS data from client.
    
    Args:
        server_socket (socket): Connected client socket
    
    Returns:
        tuple: (GPS coordinates, image data) or None if transmission ends
    """
    # Receive GPS data
    gps_data = server_socket.recv(1024).decode('utf-8')
    
    # Check for end of transmission
    if gps_data == "END":
        return None
    
    # Acknowledge GPS data receipt
    server_socket.sendall(b"GPS_RECEIVED")
    
    # Receive image size
    size_data = server_socket.recv(4)
    image_size = struct.unpack("!I", size_data)[0]
    
    # Acknowledge size receipt
    server_socket.sendall(b"SIZE_RECEIVED")
    
    # Receive image data
    image_data = b''
    while len(image_data) < image_size:
        chunk = server_socket.recv(min(4096, image_size - len(image_data)))
        if not chunk:
            break
        image_data += chunk
    
    # Parse GPS coordinates
    if gps_data != "None":
        # Split coordinates
        coord_parts = gps_data.split(',')
        
        # If more than two parts, it might be a full coordinate string
        if len(coord_parts) > 2:
            lat_str = coord_parts[0]
            lon_str = coord_parts[1] if len(coord_parts) >= 2 else None
        else:
            # Assume it's a standard comma-separated lat,lon format
            lat_str, lon_str = coord_parts
        
        # Parse coordinates
        lat = parse_gps_coordinate(lat_str)
        lon = parse_gps_coordinate(lon_str)
        
        # Validate parsed coordinates
        if lat is None or lon is None:
            print(f"Invalid GPS coordinates: {gps_data}")
            return None
    else:
        # Optional: assign default coordinates or skip
        return None
    
    return (lat, lon, image_data)

def main():
    # Server configuration
    HOST = 'localhost'  # Listen on all available interfaces
    PORT = 65432        # Port to listen on
    
    # Create map instance
    image_map = ImageMap()
    
    # Set up server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        # Accept client connection
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            
            while True:
                # Receive image
                image_info = receive_image(conn)
                
                # Check for end of transmission
                if image_info is None:
                    break
                
                # Add image to map
                lat, lon, image_data = image_info
                image_map.add_image_marker(lat, lon, image_data)
    
    print("Image mapping completed!")

if __name__ == "__main__":
    main()