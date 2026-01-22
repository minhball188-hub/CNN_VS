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

# ==========================================
# EXPORT FUNCTIONS (HYBRID & SMART)
# ==========================================
from skimage.measure import approximate_polygon

def apply_hybrid_processing(contour_pixels, pixel_size_x, pixel_size_y, R_in, R_out):
    """
    THUẬT TOÁN HYBRID:
    1. Chuyển đổi Pixel -> MM.
    2. Làm mượt toàn bộ (Smooth) để khử răng cưa bên trong.
    3. Xử lý biên (Snap & Overshoot): Nếu điểm nằm gần biên, ép nó ra ngoài hẳn.
    """
    # 1. Chuyển sang tọa độ thực (MM)
    # Lưu ý: contour của skimage trả về (row, col) -> (y, x)
    points_mm = []
    for row, col in contour_pixels:
        x = (col - 2) * pixel_size_x  # Trừ padding
        y = (row - 2) * pixel_size_y
        points_mm.append([x, y])
    
    # 2. LÀM MƯỢT (SMOOTHING)
    # tolerance=0.05mm: Đủ lớn để xóa răng cưa pixel (0.04mm), đủ nhỏ để giữ dáng cong
    points_array = np.array(points_mm)
    simplified_array = approximate_polygon(points_array, tolerance=0.05)
    
    # 3. XỬ LÝ BIÊN (SNAP & OVERSHOOT)
    # Mục tiêu: Đảm bảo biên cắt sạch, không để lại răng cưa ở R_out hay 0 độ
    final_points = []
    
    # Ngưỡng để nhận diện điểm biên (dựa trên sai số làm mượt + pixel)
    # Nếu điểm cách biên < 0.8mm thì coi như là điểm biên -> Đẩy lố ra
    BOUNDARY_THRESHOLD = 0.8 
    OVERSHOOT_VAL = 1.0 # Đẩy ra ngoài 1mm để cắt cho ngọt
    
    for p in simplified_array:
        x, y = p[0], p[1]
        r = np.sqrt(x**2 + y**2)
        
        new_x, new_y = x, y
        is_boundary_point = False
        
        # --- Check R_out (Biên ngoài) ---
        if abs(r - R_out) < BOUNDARY_THRESHOLD:
            # Đẩy bán kính ra R_out + 1mm
            ratio = (R_out + OVERSHOOT_VAL) / r
            new_x *= ratio
            new_y *= ratio
            is_boundary_point = True
            
        # --- Check R_in (Biên trong - nếu có) ---
        elif abs(r - R_in) < BOUNDARY_THRESHOLD:
            # Đẩy bán kính vào R_in - 1mm
            ratio = (R_in - OVERSHOOT_VAL) / r
            new_x *= ratio
            new_y *= ratio
            is_boundary_point = True

        # --- Check Góc 0 độ (Trục X) ---
        # Nếu y rất nhỏ -> Đẩy y xuống âm
        if abs(y) < BOUNDARY_THRESHOLD and x > 0:
            new_y = -OVERSHOOT_VAL
            is_boundary_point = True
            
        # --- Check Góc 90 độ (Trục Y) ---
        # Nếu x rất nhỏ -> Đẩy x xuống âm (sang trái)
        if abs(x) < BOUNDARY_THRESHOLD and y > 0:
            new_x = -OVERSHOOT_VAL
            is_boundary_point = True
            
        final_points.append((new_x, new_y))
        
    return final_points

def export_air_with_layers(binary_mask, pixel_size_x, pixel_size_y, msp):
    """
    Export Air với chiến thuật Hybrid:
    - Vỏ ngoài (CUT_GROUP): Overshoot mạnh để cắt biên.
    - Đảo trong (KEEP_GROUP): Chỉ làm mượt, giữ nguyên vị trí.
    """
    labeled, num_features = ndimage.label(binary_mask)
    count_cut = 0
    count_keep = 0
    all_contours = []
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        if len(contours) == 0: continue
        
        # Sắp xếp: dài nhất = outer (vỏ), còn lại = inner (đảo)
        contours_sorted = sorted(contours, key=len, reverse=True)
        
        for i, contour in enumerate(contours_sorted):
            # === ÁP DỤNG HYBRID PROCESSING ===
            # Bước này biến Pixel -> Vector mượt -> Vector sạch biên
            processed_points = apply_hybrid_processing(contour, pixel_size_x, pixel_size_y, R_in, R_out)
            
            if len(processed_points) < 3: continue
            
            if i == 0:
                # Vỏ ngoài: Dùng để cắt -> Layer CUT
                layer = "CUT_GROUP"
                count_cut += 1
            else:
                # Đảo bên trong: Giữ nguyên -> Layer KEEP
                # Lưu ý: Hàm apply_hybrid_processing vẫn an toàn với đảo
                # vì đảo nằm giữa, không chạm biên R_out/Angle nên sẽ không bị Overshoot
                layer = "KEEP_GROUP"
                count_keep += 1
            
            msp.add_lwpolyline(processed_points, close=True, dxfattribs={'layer': layer})
            all_contours.append(contour) # Lưu contour gốc để vẽ hình visualization
    
    return count_cut, count_keep, all_contours

def export_magnet_as_polyline(binary_mask, pixel_size_x, pixel_size_y, msp, layer_name):
    """
    Export Magnet - CÓ SIMPLIFY (Làm mượt)
    Để tránh lưới Ansys bị nát do răng cưa pixel
    """
    labeled, num_features = ndimage.label(binary_mask)
    all_contours = []
    count = 0
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        # Pad để contour không bị đứt nếu chạm biên ảnh
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        for contour in contours:
            if len(contour) < 3: continue

            # 1. Chuyển sang mm
            points_mm = []
            for row, col in contour:
                x_mm = (col - 2) * pixel_size_x
                y_mm = (row - 2) * pixel_size_y
                points_mm.append([x_mm, y_mm])
            
            # 2. QUAN TRỌNG: LÀM MƯỢT (Smoothing)
            # tolerance=0.05 giúp khử răng cưa nhưng vẫn giữ dáng chữ nhật/cong của nam châm
            # Nếu không có dòng này, Ansys sẽ "khóc thét" vì lưới quá dày!
            points_array = np.array(points_mm)
            simplified_array = approximate_polygon(points_array, tolerance=0.05)
            
            # 3. Vẽ DXF
            # Chuyển về list tuple cho ezdxf
            final_points = [(p[0], p[1]) for p in simplified_array]
            
            msp.add_lwpolyline(final_points, close=True, dxfattribs={'layer': layer_name})
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