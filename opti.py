# Optimized version of the app.py file, which uses ThreadPoolExecutor to send multiple images simultaneously.
import os
import socket
import struct
import threading
import concurrent.futures
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Shared resources with thread safety
received_images = []
received_images_lock = threading.Lock()
IMAGE_PROCESSING_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images')
def get_images():
    with received_images_lock:
        return jsonify(received_images)

def process_image(lat, lon, image_data):
    try:
        # Image processing pipeline
        image = Image.open(BytesIO(image_data))
        image.thumbnail((400, 400))
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        offset = 0.0002
        image_bounds = [
            [lat - offset, lon - offset],
            [lat + offset, lon + offset]
        ]
        
        image_entry = {
            'latitude': lat,
            'longitude': lon,
            'image': f"data:image/jpeg;base64,{img_str}",
            'bounds': image_bounds
        }
        
        # Thread-safe storage update
        with received_images_lock:
            received_images.append(image_entry)
        
        # Socket.IO emission with proper context
        with app.app_context():
            socketio.emit('new_image', image_entry)
            
    except Exception as e:
        print(f"Image processing error: {e}")

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
    with conn:
        while True:
            try:
                # Protocol handling
                gps_data = conn.recv(1024).decode('utf-8')
                if gps_data == "END":
                    break
                
                conn.sendall(b"GPS_RECEIVED")
                size_data = conn.recv(4)
                image_size = struct.unpack("!I", size_data)[0]
                conn.sendall(b"SIZE_RECEIVED")
                
                # Stream image data
                image_data = b''
                while len(image_data) < image_size:
                    chunk = conn.recv(min(4096, image_size - len(image_data)))
                    if not chunk:
                        break
                    image_data += chunk
                
                # Parallel processing
                if gps_data != "None":
                    coord_parts = gps_data.split(',')
                    lat_str = coord_parts[0]
                    lon_str = coord_parts[1] if len(coord_parts) > 1 else None
                    
                    lat = parse_gps_coordinate(lat_str)
                    lon = parse_gps_coordinate(lon_str)
                    
                    if lat and lon:
                        IMAGE_PROCESSING_POOL.submit(
                            process_image, lat, lon, image_data
                        )
                        
            except (ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"Client handling error: {e}")
                break

def receive_socket_image():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('0.0.0.0', 65432))
        server_socket.listen(100)  # Allow more pending connections
        print("Socket server listening")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as connection_pool:
            while True:
                conn, addr = server_socket.accept()
                connection_pool.submit(handle_client, conn)

def start_socket_server():
    socketio.start_background_task(receive_socket_image)

if __name__ == '__main__':
    start_socket_server()
    socketio.run(app, debug=True, port=5000, use_reloader=False)
