import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
from scipy import ndimage
import ezdxf

# ==========================================
# PHẦN 1: SINH HÌNH
# ==========================================

def ngnet_generate_shape(weights, centers, grid_x, grid_y, sigma):
    function_value = np.zeros_like(grid_x)
    sum_gaussian = np.zeros_like(grid_x)
    for w, center in zip(weights, centers):
        dist_sq = (grid_x - center[0])**2 + (grid_y - center[1])**2
        g = np.exp(-dist_sq / (2 * sigma**2))
        function_value += w * g
        sum_gaussian += g
    sum_gaussian[sum_gaussian == 0] = 1e-10
    return function_value / sum_gaussian

# --- THIẾT LẬP ---
R_out = 39; R_in = 16.0; resolution = 1000
x = np.linspace(0, R_out, resolution)
y = np.linspace(0, R_out, resolution)
GX, GY = np.meshgrid(x, y)

pixel_size_x = R_out / (resolution - 1)
pixel_size_y = R_out / (resolution - 1)

radius = np.sqrt(GX**2 + GY**2)
angle = np.arctan2(GY, GX) * 180 / np.pi
mask_rotor_region = (radius >= R_in) & (radius <= R_out) & (angle >= 0) & (angle <= 90)

# ==========================================
# SEED CHUNG
# ==========================================
SEED = 1  # Thay đổi seed này để random thiết kế khác
np.random.seed(SEED)

# ==========================================
# NAM CHÂM
# ==========================================
def generate_random_magnet_config(R_in, R_out, num_magnets=1):
    magnets = []
    for _ in range(num_magnets):
        margin = 3.0
        r_center = np.random.uniform(R_in + margin, R_out - margin)
        theta_center = np.random.uniform(2, 88)
        center_x = r_center * np.cos(np.deg2rad(theta_center))
        center_y = r_center * np.sin(np.deg2rad(theta_center))
        width = np.random.uniform(12, 20)
        thickness = np.random.uniform(2.5, 5.0)
        ang = np.random.uniform(0, 90)
        magnets.append({'center_x': center_x, 'center_y': center_y, 
                       'width': width, 'thickness': thickness, 'angle': ang})
    return magnets

def create_magnet_mask(GX, GY, center_x, center_y, width, thickness, angle, region_mask):
    dX = GX - center_x
    dY = GY - center_y
    angle_rad = np.deg2rad(angle)
    rotated_X = dX * np.cos(angle_rad) + dY * np.sin(angle_rad)
    rotated_Y = -dX * np.sin(angle_rad) + dY * np.cos(angle_rad)
    return (np.abs(rotated_X) <= width/2) & (np.abs(rotated_Y) <= thickness/2) & region_mask

magnets_config = generate_random_magnet_config(R_in, R_out, num_magnets=1)
mask_magnet = np.zeros_like(GX, dtype=bool)
for m in magnets_config:
    mask_magnet |= create_magnet_mask(GX, GY, m['center_x'], m['center_y'],
                                       m['width'], m['thickness'], m['angle'], mask_rotor_region)

print(f"Nam châm: pos=({magnets_config[0]['center_x']:.1f}, {magnets_config[0]['center_y']:.1f})")

# ==========================================
# NGNET 0-45° + MIRROR
# ==========================================
centers = []
for r in np.linspace(R_in+2, R_out-2, 6):
    for theta in np.linspace(2, 43, 6):
        centers.append([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])

weights = np.random.uniform(-1.0, 1.0, len(centers))
phi_value = ngnet_generate_shape(weights, centers, GX, GY, sigma=4.0)

# Mirror theo đường 45° - hoán đổi X và Y
# Với điểm (x,y) ở vùng 45-90°, lấy giá trị từ điểm (y,x) ở vùng 0-45°
phi_mirrored = phi_value.copy()
ang = np.arctan2(GY, GX) * 180 / np.pi
mask_45_90 = ang > 45

# Tạo grid mirror: điểm (i,j) lấy giá trị từ (j,i)
phi_mirrored[mask_45_90] = phi_value.T[mask_45_90]

# ĐẢM BẢO ĐỐI XỨNG HOÀN HẢO: Làm đối xứng air_mask sau khi threshold
# Thay vì mirror phi rồi threshold, ta threshold trước rồi mirror mask

# ==========================================
# TẠO MASK ĐỘC LẬP
# ==========================================
# Air từ NGnet
air_mask_raw = (phi_mirrored < 0) & mask_rotor_region

# ĐẢM BẢO ĐỐI XỨNG HOÀN HẢO theo đường 45°
air_mask_sym = air_mask_raw.copy()
air_mask_sym[mask_45_90] = air_mask_raw.T[mask_45_90]

# TRÀN RA NGOÀI Rin và Rout để cắt an toàn
from scipy.ndimage import binary_dilation

DILATE_MM = 0.5  # mm tràn ra
DILATE_PIXELS = int(DILATE_MM / pixel_size_x) + 1

air_dilated = binary_dilation(air_mask_sym, iterations=DILATE_PIXELS)

# Vùng ngoài biên rotor
outside_Rout = (radius > R_out) & (radius <= R_out + DILATE_MM * 2)
inside_Rin = (radius < R_in) & (radius >= R_in - DILATE_MM * 2)
outside_rotor = outside_Rout | inside_Rin

# Air = air gốc + phần tràn ra ngoài biên
air_mask = air_mask_sym | (air_dilated & outside_rotor)

print(f"Air tràn {DILATE_MM}mm ra ngoài Rin/Rout")

# Magnet - NGUYÊN
magnet_mask = mask_magnet

# Display
display_design = np.full_like(GX, -1)
display_design[air_mask] = 0
display_design[magnet_mask] = 2

print(f"Air pixels: {np.sum(air_mask)}")
print(f"Magnet pixels: {np.sum(magnet_mask)}")
print(f"Overlap pixels: {np.sum(air_mask & magnet_mask)}")

# ==========================================
# EXPORT FUNCTIONS
# ==========================================

def export_air_with_layers(binary_mask, pixel_size_x, pixel_size_y, msp):
    """
    Export Air với 2 layers - NỘI SUY PIXEL (không simplify)
    - CUT_GROUP: viền ngoài (cắt khỏi rotor)
    - KEEP_GROUP: viền lỗ bên trong (giữ lại = đảo thép)
    """
    labeled, num_features = ndimage.label(binary_mask)
    count_cut = 0
    count_keep = 0
    all_contours = []
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        if len(contours) == 0:
            continue
        
        # Sắp xếp: dài nhất = outer, còn lại = holes
        contours_sorted = sorted(contours, key=len, reverse=True)
        
        for i, contour in enumerate(contours_sorted):
            # Chuyển sang mm - KHÔNG simplify, giữ nguyên pixel
            points = []
            for row, col in contour:
                x_mm = (col - 2) * pixel_size_x
                y_mm = (row - 2) * pixel_size_y
                points.append((x_mm, y_mm))
            
            if len(points) >= 3:
                if i == 0:
                    layer = "CUT_GROUP"
                    count_cut += 1
                else:
                    layer = "KEEP_GROUP"
                    count_keep += 1
                
                msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer})
                all_contours.append(contour)
    
    return count_cut, count_keep, all_contours

def export_magnet_as_polyline(binary_mask, pixel_size_x, pixel_size_y, msp, layer_name):
    """
    Export Magnet - NỘI SUY PIXEL (không simplify)
    """
    labeled, num_features = ndimage.label(binary_mask)
    all_contours = []
    count = 0
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        for contour in contours:
            # Chuyển sang mm - KHÔNG simplify
            points = []
            for row, col in contour:
                x_mm = (col - 2) * pixel_size_x
                y_mm = (row - 2) * pixel_size_y
                points.append((x_mm, y_mm))
            
            if len(points) >= 3:
                msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
                all_contours.append(contour)
                count += 1
    
    return count, all_contours

# ==========================================
# XUẤT DXF
# ==========================================
doc = ezdxf.new()
msp = doc.modelspace()
doc.layers.add(name="CUT_GROUP", color=4)    # Cyan - cắt khỏi rotor
doc.layers.add(name="KEEP_GROUP", color=3)   # Green - giữ lại (đảo thép)
doc.layers.add(name="MAGNET", color=1)       # Red - nam châm

# Air - chia 2 layers
num_cut, num_keep, air_contours = export_air_with_layers(air_mask, pixel_size_x, pixel_size_y, msp)

# Magnet - Polyline
num_mag, mag_contours = export_magnet_as_polyline(magnet_mask, pixel_size_x, pixel_size_y, msp, "MAGNET")

filename = "rotor_design.dxf"
doc.saveas(filename)
print(f"\nFile: {filename}")
print(f"  CUT_GROUP (viền ngoài air): {num_cut}")
print(f"  KEEP_GROUP (đảo thép): {num_keep}")
print(f"  MAGNET: {num_mag}")

# ==========================================
# VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
cmap = ListedColormap(['white', 'cyan', 'gray', 'red'])
ax1.pcolormesh(GX, GY, display_design, cmap=cmap, vmin=-1, vmax=2, shading='auto')
ax1.set_title("Pixel View (Air=cyan, Magnet=đỏ)")
ax1.plot([0, R_out], [0, R_out], 'g--', lw=2)
ax1.set_aspect('equal')

ax2 = axes[1]
ax2.set_title("Contours xuất DXF")

# Biên rotor
theta_arc = np.linspace(0, np.pi/2, 100)
ax2.plot(R_out * np.cos(theta_arc), R_out * np.sin(theta_arc), 'k-', lw=1)
ax2.plot(R_in * np.cos(theta_arc), R_in * np.sin(theta_arc), 'k-', lw=1)
ax2.plot([R_in, R_out], [0, 0], 'k-', lw=1)
ax2.plot([0, 0], [R_in, R_out], 'k-', lw=1)

# Air contours
for contour in air_contours:
    xs = [(c[1]-2) * pixel_size_x for c in contour] + [(contour[0][1]-2) * pixel_size_x]
    ys = [(c[0]-2) * pixel_size_y for c in contour] + [(contour[0][0]-2) * pixel_size_y]
    ax2.plot(xs, ys, 'b-', lw=1)

# Magnet contours
for contour in mag_contours:
    xs = [(c[1]-2) * pixel_size_x for c in contour] + [(contour[0][1]-2) * pixel_size_x]
    ys = [(c[0]-2) * pixel_size_y for c in contour] + [(contour[0][0]-2) * pixel_size_y]
    ax2.plot(xs, ys, 'r-', lw=1.5)

ax2.set_aspect('equal')
ax2.set_xlim(-2, R_out + 2)
ax2.set_ylim(-2, R_out + 2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rotor_design.png", dpi=200)
print(f"Đã lưu: rotor_design.png")
plt.show()