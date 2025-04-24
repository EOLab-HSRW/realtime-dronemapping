## Repository Structure

```
.
├── app.py             # Flask + SocketIO real-time server
├── opti.py            # Optimized server with ThreadPoolExecutor
├── main.py            # Static Folium map generator
├── image_sender.py    # Client for sending images with EXIF GPS
├── overlay.py         # GeoTIFF overlay script (Rasterio)
├── stitch.py          # Stitching script
├── requirements.txt   # Python dependencies
└── templates/
    └── index.html     # Leaflet-based real-time mapping UI
```

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/azamzubairi/realtime-dronemapping.git
   cd realtime-dronemapping
   ```
2. **Create a virtual environment & install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Quick Start

### Real-Time Streaming

1. **Start the server**  
   ```bash
   python app.py
   ```
2. **Run the client** (ensure you have an `images/` folder with JPEGs)  
   ```bash
   python image_sender.py
   ```
3. **Open your browser** at http://localhost:5000 to see live image overlays.
