import os
import cv2
import numpy as np
import random
# ──────────────────────────────────────────────#
# Importar SAM
from segment_anything import SamPredictor, sam_model_registry
# ──────────────────────────────────────────────#
# Configuración de rutas
objects_dir = "home-dataset/images/"
labels_dir = "home-dataset/labels/"
segmented_dir = "home-dataset/segmented"
backgrounds_dir = "dataset/backgrounds/"
output_images_dir = "dataset/synthetic/images/"
output_labels_dir = "dataset/synthetic/labels/"
contrastive_images_dir = "dataset/synthetic/contrastive/images/"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs(contrastive_images_dir, exist_ok=True)
os.makedirs(segmented_dir, exist_ok=True)  # Se crea la carpeta de segmentados si no existe
# ──────────────────────────────────────────────#
# Función para leer label en formato YOLO y retornar la bbox en píxeles
def read_yolo_label(label_path, img_width, img_height):
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 5:
        return None
    _, x_center, y_center, width, height = parts
    x_center = float(x_center) * img_width
    y_center = float(y_center) * img_height
    width = float(width) * img_width
    height = float(height) * img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max
# ──────────────────────────────────────────────#
# Segmentación con SAM utilizando el recorte original (sin márgenes adicionales)
def segment_object_with_sam(img, bbox, predictor):
    x_min, y_min, x_max, y_max = bbox
    predictor.set_image(img)
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits = predictor.predict(box=input_box, multimask_output=True)
    best_mask = masks[np.argmax(scores)]
    ys, xs = np.where(best_mask)
    if len(xs) == 0 or len(ys) == 0:
        print("Segmentación falló, máscara vacía para bbox:", bbox)
        return None, None
    seg_x_min, seg_y_min = xs.min(), ys.min()
    seg_x_max, seg_y_max = xs.max(), ys.max()
    seg_crop = img[seg_y_min:seg_y_max + 1, seg_x_min:seg_x_max + 1]
    mask_crop = best_mask[seg_y_min:seg_y_max + 1, seg_x_min:seg_x_max + 1]
    mask_crop_uint8 = (mask_crop.astype(np.uint8)) * 255
    b, g, r = cv2.split(seg_crop)
    seg_crop_bgra = cv2.merge([b, g, r, mask_crop_uint8])
    return seg_crop_bgra, mask_crop_uint8
# ──────────────────────────────────────────────#
# Función para componer el objeto sobre el fondo utilizando blending alfa (si el objeto tiene canal alfa)
def composite_object_on_background(obj_img, mask, bg_img):
    h_obj, w_obj = obj_img.shape[:2]
    scale = random.uniform(0.25, 0.5)
    new_w = int(w_obj * scale)
    new_h = int(h_obj * scale)
    obj_img_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    h_bg, w_bg = bg_img.shape[:2]
    max_x = w_bg - new_w
    max_y = h_bg - new_h
    if max_x <= 0 or max_y <= 0:
        print("Fondo demasiado pequeño para componer el objeto.")
        return None, None
    x_offset = random.randint(0, max_x)
    y_offset = random.randint(0, max_y)
    if obj_img_resized.shape[2] == 4:
        alpha_channel = obj_img_resized[:, :, 3] / 255.0
        for c in range(3):
            bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                alpha_channel * obj_img_resized[:, :, c] +
                (1 - alpha_channel) * bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )
        mask_binary = (alpha_channel * 255).astype(np.uint8)
    else:
        mask_bool = mask_resized.astype(bool)
        roi = bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        roi[mask_bool] = obj_img_resized[mask_bool]
        bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi
        mask_binary = (mask_resized > 0).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask_binary)
    x_min = x + x_offset
    y_min = y + y_offset
    x_max = x_min + w
    y_max = y_min + h
    return bg_img, (x_min, y_min, x_max, y_max)
# ──────────────────────────────────────────────#
# Convertir bbox a formato YOLO
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    return x_center, y_center, bbox_width, bbox_height
# ──────────────────────────────────────────────#
# Funciones de Data Augmentation
def generate_zoom_variants(seg_obj, seg_mask):
    variants = []
    h, w = seg_obj.shape[:2]
    # Zoom In
    for i in range(2):
        factor = random.uniform(1.1, 1.4)
        new_w = int(w / factor)
        new_h = int(h / factor)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        crop_img = seg_obj[start_y:start_y+new_h, start_x:start_x+new_w]
        crop_mask = seg_mask[start_y:start_y+new_h, start_x:start_x+new_w]
        zoom_in_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
        zoom_in_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        variants.append((zoom_in_img, zoom_in_mask))
    # Zoom Out
    for i in range(2):
        factor = random.uniform(0.5, 0.9)
        new_w = int(w * factor)
        new_h = int(h * factor)
        zoom_out_img = cv2.resize(seg_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        zoom_out_mask = cv2.resize(seg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros_like(seg_obj)
        canvas_mask = np.zeros_like(seg_mask)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = zoom_out_img
        canvas_mask[start_y:start_y+new_h, start_x:start_x+new_w] = zoom_out_mask
        variants.append((canvas, canvas_mask))
    return variants
def apply_illumination_and_blur(img):
    has_alpha = (img.shape[2] == 4)
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
    variants = []
    # Variante 1: Brillo aumentado + blur suave
    alpha_val = random.uniform(1.0, 1.2)
    beta_val = random.randint(10, 30)
    bright = cv2.convertScaleAbs(bgr, alpha=alpha_val, beta=beta_val)
    ksize = random.choice([3,5])
    bright_blur = cv2.GaussianBlur(bright, (ksize,ksize), 0)
    if has_alpha:
        bright_blur = cv2.cvtColor(bright_blur, cv2.COLOR_BGR2BGRA)
        bright_blur[:, :, 3] = alpha
    variants.append(bright_blur)
    # Variante 2: Brillo reducido + blur moderado
    alpha_val = random.uniform(0.8, 1.0)
    beta_val = random.randint(-30,-10)
    dark = cv2.convertScaleAbs(bgr, alpha=alpha_val, beta=beta_val)
    ksize = random.choice([5,7])
    dark_blur = cv2.GaussianBlur(dark, (ksize,ksize), 0)
    if has_alpha:
        dark_blur = cv2.cvtColor(dark_blur, cv2.COLOR_BGR2BGRA)
        dark_blur[:, :, 3] = alpha
    variants.append(dark_blur)
    # Variante 3: Contraste aumentado + blur suave
    alpha_val = random.uniform(1.2, 1.5)
    beta_val = 0
    contrast = cv2.convertScaleAbs(bgr, alpha=alpha_val, beta=beta_val)
    ksize = random.choice([3,5])
    contrast_blur = cv2.GaussianBlur(contrast, (ksize,ksize), 0)
    if has_alpha:
        contrast_blur = cv2.cvtColor(contrast_blur, cv2.COLOR_BGR2BGRA)
        contrast_blur[:, :, 3] = alpha
    variants.append(contrast_blur)
    return variants
def generate_illumination_variants(obj_img, mask):
    variants = []
    variants.append((obj_img.copy(), mask.copy()))
    mods = apply_illumination_and_blur(obj_img)
    for mod in mods:
        variants.append((mod, mask.copy()))
    return variants
def apply_perspective_transform(img, mode):
    h, w = img.shape[:2]
    src_points = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst_points = get_perspective_points(w, h, mode)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    transformed = cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return transformed
def apply_perspective_transform_mask(mask, mode):
    h, w = mask.shape[:2]
    src_points = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst_points = get_perspective_points(w, h, mode)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(mask, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed
def get_perspective_points(w, h, mode):
    src_points = np.float32([[0,0],[w,0],[w,h],[0,h]])
    if mode == "original":
        return src_points.copy()
    if random.random() < 0.5:
        magnitude = random.uniform(0.05, 0.2) * min(w,h)
    else:
        magnitude = random.uniform(0.1, 0.4) * min(w,h)
    angle_ranges = {
        "top_right": (280,350),
        "top_left": (190,260),
        "right": (-20,20),
        "left": (160,200),
        "top": (230,310),
        "bottom": (50,130),
        "bottom_right": (0,80),
        "bottom_left": (100,180)
    }
    if mode in angle_ranges:
        low, high = angle_ranges[mode]
        angle = random.uniform(low, high)
    else:
        angle = 0.0
    angle_rad = np.deg2rad(angle)
    offset = np.array([np.cos(angle_rad)*magnitude, np.sin(angle_rad)*magnitude])
    if mode == "bottom_right":
        weights = [1.0, 0.5, 0.0, 0.5]
    elif mode == "bottom_left":
        weights = [0.5, 1.0, 0.5, 0.0]
    elif mode == "top_right":
        weights = [0.5, 0.0, 0.5, 1.0]
    elif mode == "top_left":
        weights = [0.0, 0.5, 1.0, 0.5]
    elif mode == "right":
        weights = [1.0, 0.0, 0.0, 1.0]
    elif mode == "left":
        weights = [0.0, 1.0, 1.0, 0.0]
    elif mode == "top":
        weights = [0.0, 0.0, 1.0, 1.0]
    elif mode == "bottom":
        weights = [1.0, 1.0, 0.0, 0.0]
    else:
        weights = [0,0,0,0]
    dst_points = []
    for i, pt in enumerate(src_points):
        new_pt = pt + weights[i]*offset
        dst_points.append(new_pt)
    return np.float32(dst_points)
def get_perspective_variants(segmented_obj, seg_mask):
    modes = ["original", "top_right", "top_left", "right", "left", "top", "bottom", "bottom_right", "bottom_left"]
    variants = {}
    for mode in modes:
        if mode == "original":
            if segmented_obj.shape[2] == 3:
                bgra = cv2.cvtColor(segmented_obj, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = seg_mask
                variants[mode] = (bgra, seg_mask.copy())
            else:
                variants[mode] = (segmented_obj.copy(), seg_mask.copy())
        else:
            transformed_obj = apply_perspective_transform(segmented_obj, mode)
            transformed_mask = apply_perspective_transform_mask(seg_mask, mode)
            variants[mode] = (transformed_obj, transformed_mask)
    return variants
def composite_side_by_side(bg_img, obj_img, mask, num_copies):
    h_bg, w_bg = bg_img.shape[:2]
    h_obj, w_obj = obj_img.shape[:2]
    scale = random.uniform(0.25, 0.5)
    new_w = int(w_obj*scale)
    new_h = int(h_obj*scale)
    total_width = num_copies*new_w
    if total_width > w_bg or new_h > h_bg:
        scale = min(w_bg/(num_copies*w_obj), h_bg/h_obj)*0.9
        new_w = int(w_obj*scale)
        new_h = int(h_obj*scale)
        total_width = num_copies*new_w
    x_offset = random.randint(0, w_bg-total_width)
    y_offset = random.randint(0, h_bg-new_h)
    obj_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    comp_img = bg_img.copy()
    for i in range(num_copies):
        xi = x_offset + i*new_w
        yi = y_offset
        if obj_resized.shape[2] == 4:
            alpha_channel = obj_resized[:, :, 3] / 255.0
            for c in range(3):
                comp_img[yi:yi+new_h, xi:xi+new_w, c] = (
                    alpha_channel*obj_resized[:, :, c] +
                    (1-alpha_channel)*comp_img[yi:yi+new_h, xi:xi+new_w, c]
                )
        else:
            mask_bool = mask_resized.astype(bool)
            roi = comp_img[yi:yi+new_h, xi:xi+new_w]
            roi[mask_bool] = obj_resized[mask_bool]
            comp_img[yi:yi+new_h, xi:xi+new_w] = roi
        boxes.append((xi, yi, xi+new_w, yi+new_h))
    return comp_img, boxes
def composite_random(bg_img, obj_img, mask, num_copies, max_attempts=50):
    h_bg, w_bg = bg_img.shape[:2]
    h_obj, w_obj = obj_img.shape[:2]
    scale = random.uniform(0.25,0.5)
    new_w = int(w_obj*scale)
    new_h = int(h_obj*scale)
    if new_w > w_bg or new_h > h_bg:
        scale = min(w_bg/w_obj, h_bg/h_obj)*0.9
        new_w = int(w_obj*scale)
        new_h = int(h_obj*scale)
    obj_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    comp_img = bg_img.copy()
    boxes = []
    attempts = 0
    placed = 0
    perspective_modes = ["top_right", "top_left", "right", "left", "top", "bottom", "bottom_right", "bottom_left"]
    while placed < num_copies and attempts < max_attempts*num_copies:
        attempts += 1
        x_rand = random.randint(0, w_bg-new_w)
        y_rand = random.randint(0, h_bg-new_h)
        candidate_box = (x_rand, y_rand, x_rand+new_w, y_rand+new_h)
        overlap = False
        for b in boxes:
            if (x_rand >= b[2] or b[0] >= x_rand+new_w or y_rand >= b[3] or b[1] >= y_rand+new_h):
                continue
            else:
                overlap = True
                break
        if overlap:
            continue
        random_mode = random.choice(perspective_modes)
        obj_transformed = apply_perspective_transform(obj_resized, random_mode)
        mask_transformed = apply_perspective_transform_mask(mask_resized, random_mode)
        if obj_transformed.shape[2] == 4:
            alpha_channel = obj_transformed[:, :, 3] / 255.0
            for c in range(3):
                comp_img[y_rand:y_rand+new_h, x_rand:x_rand+new_w, c] = (
                    alpha_channel*obj_transformed[:, :, c] +
                    (1-alpha_channel)*comp_img[y_rand:y_rand+new_h, x_rand:x_rand+new_w, c]
                )
        else:
            mask_bool = mask_transformed.astype(bool)
            roi = comp_img[y_rand:y_rand+new_h, x_rand:x_rand+new_w]
            roi[mask_bool] = obj_transformed[mask_bool]
            comp_img[y_rand:y_rand+new_h, x_rand:x_rand+new_w] = roi
        boxes.append(candidate_box)
        placed += 1
    return comp_img, boxes
# ──────────────────────────────────────────────#
# Función apply_random_occlusion agregada para integrar las nuevas oclusiones
def apply_random_occlusion(obj_img, seg_mask, occlusion_prob=0.5):
    if random.random() > occlusion_prob:
        return obj_img, seg_mask
    if random.random() < 0.5:
        return apply_grid_occlusion(obj_img, seg_mask)
    else:
        return apply_triangle_occlusion(obj_img, seg_mask)
# ──────────────────────────────────────────────#
def generate_contrastive_pair(segmented_obj, seg_mask):
    persp_modes = ["top_right", "top_left", "right", "left", "top", "bottom", "bottom_right", "bottom_left"]
    # Versión 1
    zoomed_versions = generate_zoom_variants(segmented_obj, seg_mask)
    version1, mask1 = random.choice(zoomed_versions)
    version1, mask1 = random.choice(generate_illumination_variants(version1, mask1))
    version1 = apply_perspective_transform(version1, random.choice(persp_modes))
    mask1 = apply_perspective_transform_mask(mask1, random.choice(persp_modes))
    version1, mask1 = apply_random_occlusion(version1, mask1)
    # Versión 2
    zoomed_versions2 = generate_zoom_variants(segmented_obj, seg_mask)
    version2, mask2 = random.choice(zoomed_versions2)
    version2, mask2 = random.choice(generate_illumination_variants(version2, mask2))
    version2 = apply_perspective_transform(version2, random.choice(persp_modes))
    mask2 = apply_perspective_transform_mask(mask2, random.choice(persp_modes))
    version2, mask2 = apply_random_occlusion(version2, mask2)
    return (version1, mask1), (version2, mask2)
def apply_background_noise(bg_img, noise_level=0.05):
    noise = np.random.randint(0, int(255*noise_level), bg_img.shape, dtype=np.uint8)
    noisy_bg = cv2.add(bg_img, noise)
    return noisy_bg
# ──────────────────────────────────────────────#
# Funciones de oclusión inteligentes
def crop_to_visible(obj_img, seg_mask):
    """
    Calcula el bounding box de los píxeles visibles (donde seg_mask > 0)
    y recorta la imagen y la máscara a ese rectángulo.
    """
    ys, xs = np.where(seg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return obj_img, seg_mask
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped_img = obj_img[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = seg_mask[y_min:y_max+1, x_min:x_max+1]
    if cropped_img.shape[2] == 4:
        cropped_img[:, :, 3] = cropped_mask
    return cropped_img, cropped_mask
def apply_grid_occlusion(obj_img, seg_mask):
    """
    En vez de pintar cuadros de color para ocluir, se “ocultan” (pone a 0 en la máscara)
    determinadas regiones y luego se recorta la imagen al área visible.
    """
    h, w = obj_img.shape[:2]
    regions = [
        (0, 0, w, int(h/4)),
        (0, int(3*h/4), w, h),
        (0, 0, int(w/4), h),
        (int(3*w/4), 0, w, h),
        (0, 0, w, int(h/3)),
        (0, int(2*h/3), w, h),
        (0, 0, int(w/3), h),
        (int(2*w/3), 0, w, h)
    ]
    chosen_regions = []
    for region in regions:
        if random.random() < 0.5:
            chosen_regions.append(region)
    if not chosen_regions:
        chosen_regions.append(random.choice(regions))
    for (x_min, y_min, x_max, y_max) in chosen_regions:
        seg_mask[y_min:y_max, x_min:x_max] = 0
    cropped_img, cropped_mask = crop_to_visible(obj_img, seg_mask)
    return cropped_img, cropped_mask
def apply_triangle_occlusion(obj_img, seg_mask):
    """
    Aplica oclusiones con forma de triángulo y luego recorta la imagen y la máscara
    al área visible.
    """
    h, w = obj_img.shape[:2]
    add_left = random.random() < 0.5
    add_right = random.random() < 0.5
    if not add_left and not add_right:
        add_left = True
    if add_left:
        theta = random.uniform(5,85)
        L_candidate = int(h / np.tan(np.deg2rad(theta)))
        L_candidate = int(np.clip(L_candidate, 0.25*w, 0.5*w))
        pts_left = np.array([[0, h], [L_candidate, h], [0, 0]], np.int32).reshape((-1,1,2))
        cv2.fillPoly(seg_mask, [pts_left], 0)
    if add_right:
        theta = random.uniform(5,85)
        L_candidate = int(h / np.tan(np.deg2rad(theta)))
        L_candidate = int(np.clip(L_candidate, 0.25*w, 0.5*w))
        pts_right = np.array([[w, h], [w - L_candidate, h], [w, 0]], np.int32).reshape((-1,1,2))
        cv2.fillPoly(seg_mask, [pts_right], 0)
    if random.random() < 0.5:
        y_start = int(3*h/4)
        seg_mask[y_start:h, 0:w] = 0
    cropped_img, cropped_mask = crop_to_visible(obj_img, seg_mask)
    return cropped_img, cropped_mask
# ──────────────────────────────────────────────#
# Función principal
def main():
    count = 0
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    background_files = [f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not background_files:
        print("No se encontraron imágenes de fondo en:", backgrounds_dir)
        return
    object_files = [f for f in os.listdir(objects_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not object_files:
        print("No se encontraron imágenes de objeto en:", objects_dir)
        return
    print("Fondos encontrados:", len(background_files))
    print("Objetos encontrados:", len(object_files))
    segmented_objects = []
    if os.path.exists(segmented_dir) and len(os.listdir(segmented_dir)) > 0:
        print("Cargando objetos segmentados desde:", segmented_dir)
        seg_files = [f for f in os.listdir(segmented_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        for seg_file in seg_files:
            seg_path = os.path.join(segmented_dir, seg_file)
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg_img is None:
                print("No se pudo cargar:", seg_file)
                continue
            if seg_img.shape[2] == 4:
                seg_mask = seg_img[:, :, 3]
            else:
                seg_mask = np.ones(seg_img.shape[:2], dtype=np.uint8)*255
            segmented_objects.append((seg_img, seg_mask))
            print(f"Cargado objeto segmentado: {seg_file}")
    else:
        print("No se encontró carpeta de segmentados o está vacía, procediendo a segmentar imágenes.")
        for obj_file in object_files:
            base_name, _ = os.path.splitext(obj_file)
            label_file = base_name + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            obj_path = os.path.join(objects_dir, obj_file)
            obj_full = cv2.imread(obj_path)
            if obj_full is None:
                print(f"No se pudo leer la imagen: {obj_file}")
                continue
            h_obj_img, w_obj_img = obj_full.shape[:2]
            if not os.path.exists(label_path):
                print(f"Label no encontrado para {obj_file}")
                continue
            bbox = read_yolo_label(label_path, w_obj_img, h_obj_img)
            if bbox is None:
                print(f"Formato de label incorrecto en: {label_file}")
                continue
            segmented_obj, seg_mask = segment_object_with_sam(obj_full, bbox, predictor)
            if segmented_obj is None or seg_mask is None:
                print(f"Segmentación falló para {obj_file}")
                continue
            out_seg_path = os.path.join(segmented_dir, base_name+".png")
            cv2.imwrite(out_seg_path, segmented_obj)
            segmented_objects.append((segmented_obj, seg_mask))
            print(f"Segmentado y guardado: {obj_file}")
    print("Objetos segmentados correctamente:", len(segmented_objects))
    if not segmented_objects:
        print("No se pudo segmentar ningún objeto.")
        return
    contrastive_pairs_file = os.path.join("dataset/synthetic/contrastive", "pairs.txt")
    with open(contrastive_pairs_file, "w") as cp_file:
        for idx, (segmented_obj, seg_mask) in enumerate(segmented_objects):
            (v1, m1), (v2, m2) = generate_contrastive_pair(segmented_obj, seg_mask)
            file1 = f"contrastive_{idx:05d}_1.png"
            file2 = f"contrastive_{idx:05d}_2.png"
            cv2.imwrite(os.path.join(contrastive_images_dir, file1), v1)
            cv2.imwrite(os.path.join(contrastive_images_dir, file2), v2)
            cp_file.write(f"{file1},{file2}\n")
            print(f"Par contrastivo generado: {file1}, {file2}")
    for bg_file in background_files:
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_original = cv2.imread(bg_path)
        if bg_original is None:
            print(f"No se pudo leer el fondo: {bg_file}")
            continue
        bg_original = cv2.resize(bg_original, (640,640))
        if random.random() < 0.3:
            bg_original = apply_background_noise(bg_original)
        print(f"Procesando fondo: {bg_file}")
        for (segmented_obj, seg_mask) in segmented_objects:
            zoom_variants = generate_zoom_variants(segmented_obj, seg_mask)
            illumination_variants = []
            for zoom_img, zoom_mask in zoom_variants:
                illum_vars = generate_illumination_variants(zoom_img, zoom_mask)
                illumination_variants.extend(illum_vars)
            all_perspective_variants = []
            for (obj_img, mask_img) in illumination_variants:
                perspective_dict = get_perspective_variants(obj_img, mask_img)
                for mode, (p_obj, p_mask) in perspective_dict.items():
                    all_perspective_variants.append((mode, p_obj, p_mask))
            print(f"Generadas {len(all_perspective_variants)} variantes para un objeto.")
            for mode, obj_variant, mask_variant in all_perspective_variants:
                obj_variant, mask_variant = apply_random_occlusion(obj_variant, mask_variant)
                comp_img, comp_box = composite_object_on_background(obj_variant.copy(), mask_variant.copy(), bg_original.copy())
                if comp_img is None or comp_box is None:
                    continue
                h_comp, w_comp = comp_img.shape[:2]
                yolo_box = convert_bbox_to_yolo(comp_box, w_comp, h_comp)
                out_img_name = f"synthetic_{count:05d}_{mode}.jpg"
                out_label_name = f"synthetic_{count:05d}_{mode}.txt"
                cv2.imwrite(os.path.join(output_images_dir, out_img_name), comp_img)
                with open(os.path.join(output_labels_dir, out_label_name), "w") as f:
                    f.write(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
                count += 1
                print(f"Generada imagen: {out_img_name}")
                if random.random() < 0.2:
                    X = random.randint(1,5)
                    for i in range(1, X+1):
                        num_copies = i+1
                        comp_side, boxes_side = composite_side_by_side(bg_original.copy(), obj_variant, mask_variant, num_copies)
                        if comp_side is None or not boxes_side:
                            continue
                        h_side, w_side = comp_side.shape[:2]
                        lines = []
                        for box in boxes_side:
                            yolo_box = convert_bbox_to_yolo(box, w_side, h_side)
                            lines.append(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                        out_img_name = f"synthetic_{count:05d}_{mode}_side.jpg"
                        out_label_name = f"synthetic_{count:05d}_{mode}_side.txt"
                        cv2.imwrite(os.path.join(output_images_dir, out_img_name), comp_side)
                        with open(os.path.join(output_labels_dir, out_label_name), "w") as f:
                            f.write("\n".join(lines))
                        count += 1
                        print(f"Generada composición side-by-side: {out_img_name}")
                    for i in range(1, 2*(X+1)):
                        num_copies = i+1
                        mod_obj = obj_variant.copy()
                        mod_mask = mask_variant.copy()
                        mod_obj, mod_mask = random.choice(generate_zoom_variants(mod_obj, mod_mask))
                        mod_obj, mod_mask = random.choice(generate_illumination_variants(mod_obj, mod_mask))
                        persp_modes = ["top_right", "top_left", "right", "left", "top", "bottom", "bottom_right", "bottom_left"]
                        rand_mode = random.choice(persp_modes)
                        mod_obj = apply_perspective_transform(mod_obj, rand_mode)
                        mod_mask = apply_perspective_transform_mask(mod_mask, rand_mode)
                        comp_rand, boxes_rand = composite_random(bg_original.copy(), mod_obj, mod_mask, num_copies)
                        if comp_rand is None or not boxes_rand:
                            continue
                        h_rand, w_rand = comp_rand.shape[:2]
                        lines = []
                        for box in boxes_rand:
                            yolo_box = convert_bbox_to_yolo(box, w_rand, h_rand)
                            lines.append(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                        out_img_name = f"synthetic_{count:05d}_{mode}_rand.jpg"
                        out_label_name = f"synthetic_{count:05d}_{mode}_rand.txt"
                        cv2.imwrite(os.path.join(output_images_dir, out_img_name), comp_rand)
                        with open(os.path.join(output_labels_dir, out_label_name), "w") as f:
                            f.write("\n".join(lines))
                        count += 1
                        print(f"Generada composición aleatoria: {out_img_name}")
    print(f"Generación completada. Se crearon {count} imágenes sintéticas.")
if __name__ == "__main__":
    main()
