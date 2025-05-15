import os
import sys
import time
import logging
import threading
import queue
import rasterio
import numpy as np
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import folium
from flask import Flask, send_from_directory, Response, request
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.enums import ColorInterp
import io
import base64

# --- Configuration ---
MONITOR_FOLDER = "/home/azam/Drone-Footprints/output/geotiffs/"  # Folder to watch for new .tif files
TEMP_IMAGE_DIR = "temp_pngs"      # Subdirectory to store temporary PNGs for display
MAP_HTML_FILE = "realtime_map.html" # Output HTML map file name
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
LOG_FILE = "monitor.log"
MAX_QUEUE_SIZE = 100 # Max pending updates for SSE clients

# --- Global State ---
processed_files = [] # List to store info about processed GeoTIFFs
processed_files_lock = threading.Lock() # Lock for thread-safe access to the list
update_queues = [] # List of queues for SSE clients
update_queues_lock = threading.Lock() # Lock for thread-safe access to SSE queues

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler(sys.stdout)])

# --- Flask App Setup ---
app = Flask(__name__, static_folder=None) # Disable default static folder

# --- Helper Functions ---

def get_raster_bounds_wgs84(src):
    """Get raster bounds transformed to WGS84 (EPSG:4326)"""
    try:
        # Ensure the CRS is valid
        if not src.crs:
            logging.warning(f"Source file {src.name} has no CRS defined. Skipping bounds transformation.")
            return None

        # Transform bounds to WGS84 if needed
        if src.crs.to_epsg() == 4326:
            bounds = src.bounds
            return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        else:
            wgs84_bounds = transform_bounds(src.crs, {'init': 'epsg:4326'},
                                            src.bounds.left, src.bounds.bottom,
                                            src.bounds.right, src.bounds.top)
            # Output format for folium: [[min_lat, min_lon], [max_lat, max_lon]]
            return [[wgs84_bounds[1], wgs84_bounds[0]], [wgs84_bounds[3], wgs84_bounds[2]]]
    except Exception as e:
        logging.error(f"Error transforming bounds for {src.name}: {e}")
        return None


def create_preview_image(src, max_dim=1024):
    """Creates a preview PNG image from raster data as base64 string."""
    try:
        # Determine optimal downsampling factor
        width, height = src.width, src.height
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height

        # Read data with downsampling
        out_shape = (src.count, new_height, new_width)
        img_data = src.read(out_shape=out_shape, resampling=Resampling.bilinear)

        # Handle different band types for visualization
        if src.count == 1: # Single band (grayscale)
            # Normalize data to 0-255
            nodata = src.nodata
            if nodata is not None:
                img_data = np.ma.masked_equal(img_data, nodata)

            min_val, max_val = np.nanmin(img_data), np.nanmax(img_data)
            if min_val == max_val: # Avoid division by zero if flat
                 # Handle constant image: map to mid-gray or white/black based on value
                 normalized_data = np.full(img_data.shape, 128 if min_val != 0 else 0, dtype=np.uint8)
            else:
                normalized_data = ((img_data.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # Handle potential masked values (nodata) after normalization
            if isinstance(normalized_data, np.ma.MaskedArray):
                 # Create RGBA image, set alpha to 0 for nodata
                 rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                 rgba[:, :, 0] = normalized_data[0] # R
                 rgba[:, :, 1] = normalized_data[0] # G
                 rgba[:, :, 2] = normalized_data[0] # B
                 rgba[:, :, 3] = 255 # Alpha
                 rgba[normalized_data.mask[0]] = [0, 0, 0, 0] # Make nodata transparent
                 img = Image.fromarray(rgba, 'RGBA')
            else:
                 img = Image.fromarray(normalized_data[0], 'L') # Grayscale

        elif src.count >= 3: # Multi-band (assume RGB or RGBA)
            # Check color interpretation
            colorinterp = [src.colorinterp[i] for i in range(src.count)]
            if ColorInterp.red in colorinterp and ColorInterp.green in colorinterp and ColorInterp.blue in colorinterp:
                r_idx = colorinterp.index(ColorInterp.red)
                g_idx = colorinterp.index(ColorInterp.green)
                b_idx = colorinterp.index(ColorInterp.blue)
                bands_to_read = [r_idx + 1, g_idx + 1, b_idx + 1]

                if ColorInterp.alpha in colorinterp:
                    a_idx = colorinterp.index(ColorInterp.alpha)
                    bands_to_read.append(a_idx + 1)
                    img_data = src.read(bands_to_read, out_shape=(len(bands_to_read), new_height, new_width), resampling=Resampling.bilinear)
                    # Normalize RGB channels if needed (assuming 8-bit is common, but handle others)
                    rgb_data = img_data[:3, :, :]
                    alpha_data = img_data[3, :, :]
                else:
                    img_data = src.read(bands_to_read, out_shape=(3, new_height, new_width), resampling=Resampling.bilinear)
                    rgb_data = img_data
                    alpha_data = None # No alpha band

                # Normalize/Scale RGB to 0-255 if not already uint8
                processed_bands = []
                nodata_mask = None
                nodata = src.nodata
                if nodata is not None:
                    mask = np.any(rgb_data == nodata, axis=0) # Mask if any band matches nodata
                    nodata_mask = mask

                for i in range(3):
                    band = rgb_data[i, :, :]
                    dt = band.dtype
                    if dt != np.uint8:
                         min_val, max_val = np.nanmin(band), np.nanmax(band)
                         if min_val < 0 or max_val > 255 or dt.kind == 'f':
                             if min_val == max_val:
                                 scaled_band = np.full(band.shape, 128, dtype=np.uint8) # Mid-gray for flat
                             else:
                                 scaled_band = ((band.astype(np.float32) - min_val) / (max_val - min_val) * 255)
                             processed_bands.append(scaled_band.astype(np.uint8))
                         else:
                             processed_bands.append(band.astype(np.uint8)) # Assume it's already 0-255 range
                    else:
                         processed_bands.append(band) # Already uint8

                # Stack bands: R, G, B order
                stacked_rgb = np.stack(processed_bands, axis=-1) # Shape: (height, width, 3)

                # Combine with Alpha if present
                if alpha_data is not None:
                     # Ensure alpha is uint8
                     if alpha_data.dtype != np.uint8:
                         alpha_data = (alpha_data / np.max(alpha_data) * 255).astype(np.uint8) if np.max(alpha_data) > 0 else np.zeros_like(alpha_data, dtype=np.uint8)

                     # Apply nodata mask to alpha if necessary
                     if nodata_mask is not None:
                         alpha_data[nodata_mask] = 0 # Make nodata transparent

                     # Combine RGB + Alpha
                     stacked_rgba = np.dstack((stacked_rgb, alpha_data))
                     img = Image.fromarray(stacked_rgba, 'RGBA')

                elif nodata_mask is not None: # RGB only, but with nodata
                     # Create RGBA image and set alpha based on nodata mask
                     stacked_rgba = np.dstack((stacked_rgb, np.full((new_height, new_width), 255, dtype=np.uint8)))
                     stacked_rgba[nodata_mask] = [0, 0, 0, 0] # Make nodata transparent
                     img = Image.fromarray(stacked_rgba, 'RGBA')
                else: # Standard RGB
                     img = Image.fromarray(stacked_rgb, 'RGB')

            else: # Fallback: use first 3 bands or first band as grayscale
                logging.warning(f"Could not determine RGB bands based on color interpretation for {src.name}. Using first {'3' if src.count >=3 else '1'} band(s).")
                if src.count >= 3:
                    # Read first 3 bands and try to make RGB
                    rgb_data = img_data[:3, :, :]
                     # Basic scaling for each band (similar to RGB logic above)
                    processed_bands = []
                    for i in range(3):
                        band = rgb_data[i]
                        dt = band.dtype
                        min_val, max_val = np.nanmin(band), np.nanmax(band)
                        if min_val == max_val:
                             scaled_band = np.full(band.shape, 128, dtype=np.uint8)
                        else:
                             scaled_band = ((band.astype(np.float32) - min_val) / (max_val - min_val) * 255)
                        processed_bands.append(scaled_band.astype(np.uint8))
                    stacked_rgb = np.stack(processed_bands, axis=-1)
                    img = Image.fromarray(stacked_rgb, 'RGB')
                else: # Use first band as grayscale (same logic as single band)
                    band = img_data[0]
                    nodata = src.nodata
                    if nodata is not None: band = np.ma.masked_equal(band, nodata)
                    min_val, max_val = np.nanmin(band), np.nanmax(band)
                    if min_val == max_val: normalized_data = np.full(band.shape, 128, dtype=np.uint8)
                    else: normalized_data = ((band.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    img = Image.fromarray(normalized_data, 'L')

        else: # Handle 2 bands? Or other counts?
            logging.warning(f"Unsupported band count ({src.count}) for preview generation in {src.name}. Skipping preview.")
            return None

        # Save image to a byte buffer as PNG and encode as base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        logging.error(f"Error creating preview image for {src.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_geotiff(filepath):
    """Reads GeoTIFF, extracts info, creates preview, adds to list, notifies clients."""
    logging.info(f"Detected new GeoTIFF: {filepath}")
    try:
        # Check if already processed to handle potential duplicate events
        with processed_files_lock:
            if any(f['path'] == filepath for f in processed_files):
                logging.debug(f"File already processed: {filepath}")
                return

        with rasterio.open(filepath) as src:
            bounds_wgs84 = get_raster_bounds_wgs84(src)
            if not bounds_wgs84:
                logging.error(f"Could not get WGS84 bounds for {filepath}. Skipping.")
                return

            # Create a preview image (base64 encoded PNG)
            preview_data_url = create_preview_image(src)
            if not preview_data_url:
                 logging.error(f"Failed to create preview for {filepath}. Skipping")
                 return

            file_info = {
                'path': filepath,
                'name': os.path.basename(filepath),
                'bounds': bounds_wgs84,
                'preview_url': preview_data_url, # Use base64 directly
                'crs': str(src.crs) if src.crs else 'Undefined'
            }

            with processed_files_lock:
                processed_files.append(file_info)
                logging.info(f"Processed and added: {file_info['name']}")

            # Notify listening SSE clients
            with update_queues_lock:
                msg = f"data: New file added: {file_info['name']}\n\n"
                # Add message to all active queues, handle potential full queues
                active_queues = []
                for q in update_queues:
                    try:
                        q.put_nowait(msg)
                        active_queues.append(q)
                    except queue.Full:
                        logging.warning("An SSE client queue is full. Client might be lagging.")
                        # Keep the queue, maybe it recovers, or remove if desired
                        active_queues.append(q) # Keep it for now
                    except Exception as e:
                         logging.error(f"Error putting message in queue: {e}") # Should not happen with simple queue
                # Update the list of queues (removes any problematic ones if we chose to)
                update_queues[:] = active_queues


    except rasterio.RasterioIOError as e:
        logging.error(f"Rasterio I/O error processing {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()


def generate_map_html():
    """Generates the Folium map HTML with all processed layers."""
    logging.debug("Generating map HTML...")
    # Determine map center - use first file's center or default
    map_center = [51.497836, 6.549057] # Default center if no files yet
    zoom_start = 18
    with processed_files_lock:
        if processed_files:
            first_bounds = processed_files[0]['bounds']
            map_center = [
                (first_bounds[0][0] + first_bounds[1][0]) / 2, # Avg Lat
                (first_bounds[0][1] + first_bounds[1][1]) / 2  # Avg Lon
            ]
            zoom_start = 19 # Zoom in a bit if we have data

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB dark_matter")

    # Add Tile Layers for choice
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    # Add satellite layer (requires attribution)
    folium.TileLayer(
         tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
         attr='Esri',
         name='Esri Satellite',
         overlay=False,
         control=True
     ).add_to(m)


    # Add overlays for each processed GeoTIFF
    with processed_files_lock:
        logging.debug(f"Adding {len(processed_files)} layers to the map.")
        for file_info in processed_files:
            try:
                folium.raster_layers.ImageOverlay(
                    image=file_info['preview_url'], # Use base64 encoded PNG
                    bounds=file_info['bounds'], # WGS84 bounds
                    opacity=0.7,
                    name=f"{file_info['name']} (CRS: {file_info['crs']})",
                    interactive=True,
                    cross_origin=False, # Important for base64/data URLs
                    zindex=1 # Ensure it's above base tiles
                ).add_to(m)
                logging.debug(f"Added overlay for {file_info['name']}")
            except Exception as e:
                 logging.error(f"Error adding overlay for {file_info['name']} to map: {e}")


    # Add Layer Control
    folium.LayerControl().add_to(m)

    # Save map to HTML string
    map_html = m.get_root().render()
    logging.debug("Map HTML generated successfully.")
    return map_html


# --- Watchdog Event Handler ---
class GeoTiffHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.tif', '.tiff')):
            # Give the file system a moment to finish writing
            time.sleep(1)
            # Run processing in a separate thread to avoid blocking the watchdog
            processing_thread = threading.Thread(target=process_geotiff, args=(event.src_path,))
            processing_thread.daemon = True # Allow program to exit even if thread is running
            processing_thread.start()

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page with the iframe and SSE JavaScript."""
    # Simple HTML page that embeds the map via iframe and includes SSE logic
    # Note: Using iframe avoids issues with Folium JS potentially conflicting
    # with other JS on a more complex page.
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time GeoTIFF Map</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { margin: 0; padding: 0; font-family: sans-serif; }
            h1 { text-align: center; padding: 10px; background-color: #f0f0f0; margin: 0;}
            iframe { width: 100%; height: 85vh; border: none; }
            #status { padding: 5px; font-size: 0.9em; text-align: center; color: #555; }
        </style>
    </head>
    <body>
        <h1>Real-time GeoTIFF Map Monitor</h1>
        <div id="status">Connecting to update stream...</div>
        <iframe id="map_frame" src="/map"></iframe>

        <script>
            const mapFrame = document.getElementById('map_frame');
            const statusDiv = document.getElementById('status');
            let eventSource;

            function connectSSE() {
                console.log("Connecting to SSE stream...");
                // Add cache-busting param to SSE URL? Maybe not necessary.
                eventSource = new EventSource("/stream");

                eventSource.onopen = function() {
                    console.log("SSE Connection opened.");
                    statusDiv.textContent = "Connected. Waiting for updates...";
                    // Initial load or reload of map when SSE connects/reconnects
                    mapFrame.src = '/map?_=' + new Date().getTime();
                };

                eventSource.onmessage = function(event) {
                    console.log("Map update signal received:", event.data);
                    statusDiv.textContent = "Update received! Reloading map...";
                    // Reload the iframe to show the updated map
                    // Add cache buster to ensure reload
                    mapFrame.src = '/map?_=' + new Date().getTime();
                    // Optionally reset status after a delay
                    setTimeout(() => { statusDiv.textContent = "Connected. Waiting for updates..."; }, 2000);
                };

                eventSource.onerror = function(err) {
                    console.error("EventSource failed:", err);
                    statusDiv.textContent = "Connection lost. Attempting to reconnect...";
                    eventSource.close(); // Close the current connection
                    // Retry connection after a delay
                    setTimeout(connectSSE, 5000); // Retry every 5 seconds
                };
            }

            // Initial connection attempt
            connectSSE();

            // Keep map frame same size on resize
             window.addEventListener('resize', () => {
                 // Optionally adjust iframe height if needed, e.g., based on window height
             });

        </script>
    </body>
    </html>
    """

@app.route('/map')
def map_viewer():
    """Serves the generated Folium map HTML."""
    # Generate the map HTML on demand each time this route is hit
    # This ensures the latest processed files are included
    html_content = generate_map_html()
    return Response(html_content, mimetype='text/html')


@app.route('/stream')
def stream():
    """Server-Sent Events endpoint to notify clients of updates."""
    def event_stream():
        q = queue.Queue(MAX_QUEUE_SIZE)
        with update_queues_lock:
            update_queues.append(q)
            logging.info(f"SSE client connected. Total clients: {len(update_queues)}")

        try:
            while True:
                # Wait for a message from the processing thread
                try:
                     message = q.get(timeout=30) # Check queue, timeout helps detect disconnects
                     yield message
                except queue.Empty:
                     # Send a heartbeat comment to keep connection alive if idle
                     yield ":heartbeat\n\n"

        except GeneratorExit:
             logging.info("SSE client disconnected.")
        finally:
            # Remove the queue when the client disconnects
            with update_queues_lock:
                if q in update_queues:
                     update_queues.remove(q)
                     logging.info(f"Removed SSE client queue. Remaining clients: {len(update_queues)}")

    # Set headers for SSE
    return Response(event_stream(), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no' # Important for Nginx proxying
    })

# --- Main Execution ---

def start_flask():
    """Starts the Flask development server."""
    logging.info(f"Starting Flask server on http://{SERVER_HOST}:{SERVER_PORT}")
    # Use development server for simplicity. For production, use a proper WSGI server (like Gunicorn or Waitress).
    # Turn off reloader as we handle updates via watchdog/SSE
    # debug=False is important for production/stability
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, debug=False, use_reloader=False)

def start_watcher():
    """Starts the file system watcher."""
    if not os.path.exists(MONITOR_FOLDER):
        os.makedirs(MONITOR_FOLDER)
        logging.info(f"Created monitor folder: {MONITOR_FOLDER}")

    # Optional: Create temp image dir if it doesn't exist
    # if not os.path.exists(TEMP_IMAGE_DIR):
    #     os.makedirs(TEMP_IMAGE_DIR)
    #     logging.info(f"Created temp image folder: {TEMP_IMAGE_DIR}")

    event_handler = GeoTiffHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_FOLDER, recursive=False) # Monitor only the top level
    observer.start()
    logging.info(f"Watching folder: {MONITOR_FOLDER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Watchdog observer stopped.")
    observer.join()

if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True # Allows main thread to exit even if Flask is running
    flask_thread.start()

    # Start the file system watcher in the main thread (or another thread)
    # Running watcher in main thread makes KeyboardInterrupt handling easier
    start_watcher()

    logging.info("Script finished.")
