import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
from skimage.measure import approximate_polygon
from scipy import ndimage
from scipy.ndimage import binary_dilation
import ezdxf

# ==========================================
# THIẾT LẬP
# ==========================================
R_out = 39; R_in = 16.0; resolution = 1000
x = np.linspace(0, R_out, resolution)
y = np.linspace(0, R_out, resolution)
GX, GY = np.meshgrid(x, y)

pixel_size_x = R_out / (resolution - 1)
pixel_size_y = R_out / (resolution - 1)

radius = np.sqrt(GX**2 + GY**2)
angle = np.arctan2(GY, GX) * 180 / np.pi

mask_rotor_region = (radius >= R_in) & (radius <= R_out) & (angle >= 0) & (angle <= 90)
mask_0_45 = (angle >= 0) & (angle <= 45)
mask_45_90 = angle > 45

# ==========================================
# SEED CHUNG
# ==========================================
SEED = 100
np.random.seed(SEED)

# ==========================================
# NAM CHÂM (4 góc cố định)
# ==========================================
# 4 tọa độ góc nam châm (hình chữ nhật)
MAGNET_CORNERS = [
    (9.62938, 30.84258),
    (30.84258, 9.62938),
    (7.50806, 28.72126),
    (28.72126, 7.50806)
]

def create_magnet_mask_from_corners(GX, GY, corners, region_mask):
    """Tạo mask nam châm từ 4 góc"""
    from matplotlib.path import Path
    
    # Tạo polygon path từ 4 góc
    # Sắp xếp các góc theo thứ tự để tạo hình chữ nhật đúng
    # Góc: (9.6, 30.8), (7.5, 28.7), (28.7, 7.5), (30.8, 9.6)
    sorted_corners = [
        corners[2],  # (7.50806, 28.72126) - top-left
        corners[0],  # (9.62938, 30.84258) - top-right  
        corners[1],  # (30.84258, 9.62938) - bottom-right
        corners[3],  # (28.72126, 7.50806) - bottom-left
    ]
    
    path = Path(sorted_corners)
    
    # Kiểm tra từng điểm trong grid
    points = np.column_stack((GX.ravel(), GY.ravel()))
    mask = path.contains_points(points).reshape(GX.shape)
    
    return mask & region_mask

magnet_mask = create_magnet_mask_from_corners(GX, GY, MAGNET_CORNERS, mask_rotor_region)

print(f"Nam châm: 4 góc cố định")

# ==========================================
# NGNET - CHỈ TẠO Ở 0-45°
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

# Centers - KHÔNG giới hạn, cho phép chạm biên
centers = []
for r in np.linspace(R_in, R_out, 6):       # Bỏ +2, -2
    for theta in np.linspace(0, 45, 6):      # Bỏ 2, 43 → full 0-45
        centers.append([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])

weights = np.random.uniform(-1.0, 1.0, len(centers))
phi_value = ngnet_generate_shape(weights, centers, GX, GY, sigma=4.0)

# Air mask - tràn ra biên CHỈ KHI CHẠM BIÊN
DILATE_MM = 0.2
DILATE_ANGLE = np.degrees(DILATE_MM / ((R_in + R_out) / 2))  # Chuyển mm sang độ ở bán kính trung bình

# Vùng rotor gốc (0-45°)
mask_rotor_half = mask_rotor_region & mask_0_45

# Air gốc trong vùng rotor
air_original = (phi_value < 0) & mask_rotor_half

# Kiểm tra air có chạm các biên không
touches_Rout = np.any(air_original & (radius >= R_out - 0.5))
touches_Rin = np.any(air_original & (radius <= R_in + 0.5))
touches_angle_0 = np.any(air_original & (angle <= 1))
touches_angle_45 = np.any(air_original & (angle >= 44))

# Tạo vùng mở rộng tùy theo biên nào được chạm
r_min = R_in - DILATE_MM if touches_Rin else R_in
r_max = R_out + DILATE_MM if touches_Rout else R_out
angle_min = -DILATE_ANGLE if touches_angle_0 else 0
angle_max = 45 + DILATE_ANGLE if touches_angle_45 else 45  # Góc 45 chỉ tràn 0.5mm

mask_rotor_extended = (radius >= r_min) & (radius <= r_max) & \
                      (angle >= angle_min) & (angle <= angle_max)

# Air trong vùng mở rộng
air_mask_half = (phi_value < 0) & mask_rotor_extended

print(f"Air chạm biên: Rout={touches_Rout}, Rin={touches_Rin}, 0°={touches_angle_0}, 45°={touches_angle_45}")
print(f"Tràn: Rin={r_min:.1f}, Rout={r_max:.1f}, angle=[{angle_min:.1f}°, {angle_max:.1f}°]")

# Tạo air_mask đầy đủ để hiển thị
air_mask_full = air_mask_half.copy()
air_mask_full[mask_45_90] = air_mask_half.T[mask_45_90]
print(f"Air pixels (nửa 0-45): {np.sum(air_mask_half)}")
print(f"Magnet pixels: {np.sum(magnet_mask)}")

# ==========================================
# EXPORT FUNCTIONS
# ==========================================

def export_air_with_mirror(binary_mask_half, pixel_size_x, pixel_size_y, msp, tolerance=0.1):
    """
    Export Air: simplify ở 0-45° rồi mirror sang 45-90°
    Xuất thành 2 MẢNH RIÊNG BIỆT (không ghép)
    """
    labeled, num_features = ndimage.label(binary_mask_half)
    count_cut = 0
    count_keep = 0
    all_contours = []
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        if len(contours) == 0:
            continue
        
        contours_sorted = sorted(contours, key=len, reverse=True)
        
        for i, contour in enumerate(contours_sorted):
            # Chuyển sang mm
            points_mm = []
            for row, col in contour:
                x_mm = (col - 2) * pixel_size_x
                y_mm = (row - 2) * pixel_size_y
                points_mm.append((x_mm, y_mm))
            
            # Simplify
            points_array = np.array(points_mm)
            simplified = approximate_polygon(points_array, tolerance=tolerance)
            points_half = [(p[0], p[1]) for p in simplified]
            
            # Mirror: (x, y) → (y, x)
            points_mirrored = [(p[1], p[0]) for p in points_half]
            
            layer = "CUT_GROUP" if i == 0 else "KEEP_GROUP"
            
            # XUẤT 2 MẢNH RIÊNG BIỆT
            if len(points_half) >= 3:
                msp.add_lwpolyline(points_half, close=True, dxfattribs={'layer': layer})
                all_contours.append(points_half)
                if i == 0:
                    count_cut += 1
                else:
                    count_keep += 1
            
            if len(points_mirrored) >= 3:
                msp.add_lwpolyline(points_mirrored, close=True, dxfattribs={'layer': layer})
                all_contours.append(points_mirrored)
                if i == 0:
                    count_cut += 1
                else:
                    count_keep += 1
    
    return count_cut, count_keep, all_contours

def export_magnet_as_polyline(binary_mask, pixel_size_x, pixel_size_y, msp, layer_name, tolerance=0.5):
    """
    Export Magnet - simplify bình thường (không cần đối xứng)
    """
    labeled, num_features = ndimage.label(binary_mask)
    all_contours = []
    count = 0
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        for contour in contours:
            points_mm = []
            for row, col in contour:
                x_mm = (col - 2) * pixel_size_x
                y_mm = (row - 2) * pixel_size_y
                points_mm.append((x_mm, y_mm))
            
            points_array = np.array(points_mm)
            simplified = approximate_polygon(points_array, tolerance=tolerance)
            points = [(p[0], p[1]) for p in simplified]
            
            if len(points) >= 3:
                msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
                all_contours.append(points)
                count += 1
    
    return count, all_contours

# ==========================================
# XUẤT DXF
# ==========================================
doc = ezdxf.new()
msp = doc.modelspace()
doc.layers.add(name="CUT_GROUP", color=4)
doc.layers.add(name="KEEP_GROUP", color=3)
doc.layers.add(name="MAGNET", color=1)

# Air - simplify rồi mirror
num_cut, num_keep, air_contours = export_air_with_mirror(air_mask_half, pixel_size_x, pixel_size_y, msp)

# Magnet - simplify bình thường
num_mag, mag_contours = export_magnet_as_polyline(magnet_mask, pixel_size_x, pixel_size_y, msp, "MAGNET")

filename = "rotor_design.dxf"
doc.saveas(filename)
print(f"\nFile: {filename}")
print(f"  CUT_GROUP: {num_cut}")
print(f"  KEEP_GROUP: {num_keep}")
print(f"  MAGNET: {num_mag}")

# Kiểm tra đối xứng
for entity in msp:
    if entity.dxftype() == 'LWPOLYLINE' and entity.dxf.layer == 'CUT_GROUP':
        points = list(entity.get_points())
        on_x = [p[0] for p in points if abs(p[1]) < 0.1]
        on_y = [p[1] for p in points if abs(p[0]) < 0.1]
        if on_x and on_y:
            print(f"\nĐối xứng check:")
            print(f"  Trục X max: {max(on_x):.6f}")
            print(f"  Trục Y max: {max(on_y):.6f}")
            print(f"  Chênh lệch: {abs(max(on_x) - max(on_y)):.6f} mm")
        break

# ==========================================
# VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
display = np.full_like(GX, -1)
display[air_mask_full] = 0
display[magnet_mask] = 2
cmap = ListedColormap(['white', 'cyan', 'gray', 'red'])
ax1.pcolormesh(GX, GY, display, cmap=cmap, vmin=-1, vmax=2, shading='auto')
ax1.set_title("Pixel View")
ax1.plot([0, R_out], [0, R_out], 'g--', lw=2, label='45°')
ax1.set_aspect('equal')
ax1.legend()

ax2 = axes[1]
ax2.set_title("Contours DXF (đã simplify + mirror)")

# Biên rotor
theta_arc = np.linspace(0, np.pi/2, 100)
ax2.plot(R_out * np.cos(theta_arc), R_out * np.sin(theta_arc), 'k-', lw=1)
ax2.plot(R_in * np.cos(theta_arc), R_in * np.sin(theta_arc), 'k-', lw=1)
ax2.plot([R_in, R_out], [0, 0], 'k-', lw=1)
ax2.plot([0, 0], [R_in, R_out], 'k-', lw=1)

# Air contours
for contour in air_contours:
    xs = [p[0] for p in contour] + [contour[0][0]]
    ys = [p[1] for p in contour] + [contour[0][1]]
    ax2.plot(xs, ys, 'b-', lw=1)

# Magnet contours
for contour in mag_contours:
    xs = [p[0] for p in contour] + [contour[0][0]]
    ys = [p[1] for p in contour] + [contour[0][1]]
    ax2.plot(xs, ys, 'r-', lw=1.5)

ax2.plot([0, R_out], [0, R_out], 'g--', lw=1, alpha=0.5)
ax2.set_aspect('equal')
ax2.set_xlim(-2, R_out + 2)
ax2.set_ylim(-2, R_out + 2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rotor_design.png", dpi=200)
print(f"Đã lưu: rotor_design.png")
plt.show()