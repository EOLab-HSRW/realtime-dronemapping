import os
import socket
import struct
import threading
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Global variables to track images
received_images = []

@app.route('/')
def index():
    """Render the main map page"""
    return render_template('index.html')

@app.route('/images')
def get_images():
    """Endpoint to retrieve all received images"""
    return jsonify(received_images)

def process_image(lat, lon, image_data):
    """
    Process incoming image and store for web display
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        image_data (bytes): Image data
    """
    # Convert image to base64 for web display
    image = Image.open(BytesIO(image_data))
    
    # Resize image
    image.thumbnail((400, 400))
    
    # Save to BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Calculate bounds for overlay (fixed size around coordinates)
    offset = 0.0002  # Adjust this value to control image size on map
    image_bounds = [
        [lat - offset, lon - offset],
        [lat + offset, lon + offset]
    ]
    
    # Create image entry
    image_entry = {
        'latitude': lat,
        'longitude': lon,
        'image': f"data:image/jpeg;base64,{img_str}",
        'bounds': image_bounds
    }
    
    # Add to list and emit to connected clients
    received_images.append(image_entry)
    with app.app_context():
        socketio.emit('new_image', image_entry)

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

def handle_client(conn):
    """Handle communication with a single client"""
    with conn:
        while True:
            # Receive GPS data
            gps_data = conn.recv(1024).decode('utf-8')
            
            # Check for end of transmission
            if gps_data == "END":
                break
            
            # Acknowledge GPS data receipt
            conn.sendall(b"GPS_RECEIVED")
            
            # Receive image size
            size_data = conn.recv(4)
            image_size = struct.unpack("!I", size_data)[0]
            
            # Acknowledge size receipt
            conn.sendall(b"SIZE_RECEIVED")
            
            # Receive image data
            image_data = b''
            while len(image_data) < image_size:
                chunk = conn.recv(min(4096, image_size - len(image_data)))
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
                if lat is not None and lon is not None:
                    # Process image (this will emit to web clients)
                    process_image(lat, lon, image_data)
                else:
                    print(f"Invalid GPS coordinates: {gps_data}")

def receive_socket_image():
    """
    Socket server to receive images
    """
    # Server configuration
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 65432        # Port to listen on
    
    # Set up server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Socket server listening on {HOST}:{PORT}")
        
        while True:  # Continuous accept loop
            # Accept client connection
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            # Start new thread for each client
            threading.Thread(target=handle_client, args=(conn,)).start()

def start_socket_server():
    """Start the socket server in a background thread"""
    socketio.start_background_task(receive_socket_image)

if __name__ == '__main__':
    # Start socket server
    start_socket_server()
    
    # Run Flask app
    socketio.run(app, debug=True, port=5000, use_reloader=False)