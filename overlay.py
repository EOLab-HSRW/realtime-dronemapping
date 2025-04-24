import rasterio
from rasterio.plot import show
from rasterio.warp import transform_bounds
import folium
import folium.plugins as plugins
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO

def overlay_geotiff_on_map(geotiff_path, output_html_path=None, opacity=0.7, max_zoom=22, 
                           background_threshold=None, nodata_value=None):
    """
    Overlay a GeoTIFF on an interactive satellite map with background transparency
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the GeoTIFF file
    output_html_path : str, optional
        Path to save the HTML map (if None, will display in notebook)
    opacity : float, optional
        Opacity of the overlay (0-1)
    max_zoom : int, optional
        Maximum zoom level for the map
    background_threshold : float or list, optional
        Value(s) to treat as background and make transparent
        - For single-band: a single value
        - For RGB: a list of 3 values [r,g,b]
        - If None, uses nodata value or tries to detect automatically
    nodata_value : float, optional
        Value to treat as nodata if not specified in the GeoTIFF
    """
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        
        # Get metadata
        bands = src.count
        
        # Use file's nodata value if not specified
        if nodata_value is None:
            nodata_value = src.nodata
        
        # Handle RGB or single band case appropriately
        if bands >= 3:
            # For RGB or RGBA GeoTIFFs
            # Read the RGB bands
            r = src.read(1).astype(float)
            g = src.read(2).astype(float)
            b = src.read(3).astype(float)
            
            # Create an alpha channel (1 = opaque, 0 = transparent)
            alpha = np.ones(r.shape, dtype=np.uint8) * 255
            
            # Make background transparent based on various criteria
            if nodata_value is not None:
                # Use nodata value for transparency
                mask = (r == nodata_value) | (g == nodata_value) | (b == nodata_value)
                alpha[mask] = 0
            
            if background_threshold is not None:
                # User specified background color
                if isinstance(background_threshold, list) and len(background_threshold) >= 3:
                    # RGB threshold
                    r_val, g_val, b_val = background_threshold[:3]
                    mask = (r == r_val) & (g == g_val) & (b == b_val)
                    alpha[mask] = 0
                else:
                    # Single value threshold for all bands
                    mask = (r == background_threshold) & (g == background_threshold) & (b == background_threshold)
                    alpha[mask] = 0
            
            # If no threshold specified and no nodata value, try to detect background
            if background_threshold is None and nodata_value is None:
                # Look for black (0,0,0) or white (max,max,max) as common backgrounds
                black_mask = (r == 0) & (g == 0) & (b == 0)
                
                # Get max value based on data type
                max_val = np.iinfo(r.dtype).max if np.issubdtype(r.dtype, np.integer) else 1.0
                white_mask = (r == max_val) & (g == max_val) & (b == max_val)
                
                # Combine masks
                mask = black_mask | white_mask
                alpha[mask] = 0
            
            # Robust normalization (percentile-based to avoid outliers)
            def robust_normalize(array):
                # Only consider non-background pixels for normalization
                valid_pixels = array[alpha > 0]
                if len(valid_pixels) > 0:
                    p_low, p_high = np.percentile(valid_pixels, (2, 98))
                    array_norm = np.clip((array - p_low) / (p_high - p_low + 1e-10), 0, 1)
                    return np.uint8(array_norm * 255)
                else:
                    return np.zeros_like(array, dtype=np.uint8)
            
            # Apply normalization
            r_norm = robust_normalize(r)
            g_norm = robust_normalize(g)
            b_norm = robust_normalize(b)
            
            # Stack bands to create RGBA
            rgba = np.dstack((r_norm, g_norm, b_norm, alpha))
            
        else:
            # For single band GeoTIFFs
            data = src.read(1).astype(float)
            
            # Create an alpha channel (1 = opaque, 0 = transparent)
            alpha = np.ones(data.shape, dtype=np.uint8) * 255
            
            # Handle nodata values and background
            if nodata_value is not None:
                alpha[data == nodata_value] = 0
            
            if background_threshold is not None:
                alpha[data == background_threshold] = 0
            
            # If no threshold specified and no nodata value, try to detect background
            if background_threshold is None and nodata_value is None:
                # Common background values are 0 or the maximum value
                max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
                mask = (data == 0) | (data == max_val)
                alpha[mask] = 0
            
            # Robust normalization for single band
            valid_pixels = data[alpha > 0]
            if len(valid_pixels) > 0:
                p_low, p_high = np.nanpercentile(valid_pixels, (2, 98))
                normalized = np.clip((data - p_low) / (p_high - p_low + 1e-10), 0, 1)
                normalized = np.uint8(normalized * 255)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)
            
            # Create grayscale RGBA
            rgba = np.dstack((normalized, normalized, normalized, alpha))
        
        # Get the bounds in the raster's CRS
        bounds = src.bounds
        
        # Transform bounds to WGS84 (EPSG:4326) which is used by folium
        if src.crs != 'EPSG:4326':
            west, south, east, north = transform_bounds(src.crs, 'EPSG:4326', 
                                                       bounds.left, bounds.bottom, 
                                                       bounds.right, bounds.top)
        else:
            west, south, east, north = bounds
            
        # Create a base map centered on the GeoTIFF
        center_lat = (north + south) / 2
        center_lon = (east + west) / 2
        
        # Create map with satellite imagery and high max zoom
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=15,
                      tiles=None,
                      max_zoom=max_zoom)
        
        # Add satellite basemap
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True,
            max_zoom=max_zoom
        ).add_to(m)
        
        # Add Google Satellite as an alternative option
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True,
            max_zoom=max_zoom
        ).add_to(m)
        
        # Create a PIL Image from the NumPy array with transparency
        try:
            img = Image.fromarray(rgba, mode='RGBA')
            
            # Save the image to a BytesIO object (PNG supports transparency)
            img_data = BytesIO()
            img.save(img_data, format='PNG')
            img_data.seek(0)
            
            # Encode the image as base64
            encoded = base64.b64encode(img_data.read()).decode('utf-8')
            
            # Create image overlay
            image_overlay = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded}",
                bounds=[[south, west], [north, east]],
                opacity=opacity,
                name='GeoTIFF Overlay'
            )
            
            # Add the image overlay to the map
            image_overlay.add_to(m)
            
        except Exception as e:
            print(f"Error creating image overlay: {e}")
            # Fallback method using matplotlib for visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgba)
            plt.axis('off')
            
            # Save the figure to a BytesIO object
            img_data = BytesIO()
            plt.savefig(img_data, format='PNG', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            img_data.seek(0)
            
            # Encode the image as base64
            encoded = base64.b64encode(img_data.read()).decode('utf-8')
            
            # Create image overlay
            image_overlay = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded}",
                bounds=[[south, west], [north, east]],
                opacity=opacity,
                name='GeoTIFF Overlay'
            )
            
            # Add the image overlay to the map
            image_overlay.add_to(m)
        
        # Add a scale bar
        plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)
        
        # Add zoom level display
        plugins.MousePosition(
            position='bottomright',
            separator=' | ',
            prefix="Coordinates:",
            num_digits=6
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Save to HTML or return the map
        if output_html_path:
            m.save(output_html_path)
            print(f"Map saved to {output_html_path}")
        
        return m

# Example usage
if __name__ == "__main__":
    geotiff_path = "results/odm_orthophoto/odm_orthophoto.tif"
    output_html = "geotiff_map.html"
    
    map_with_overlay = overlay_geotiff_on_map(
        geotiff_path=geotiff_path,
        output_html_path=output_html,
        opacity=0.9,
        max_zoom=24,
        # Specify background value if known, e.g.:
        background_threshold=0,  # For a black background
        # nodata_value=-9999,      # For a specific nodata value
    )

   

