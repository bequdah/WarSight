"""
Grad-CAM XAI Engine for YOLOv8  —  Spatial Loss Version
=========================================================
الإصلاح الجذري: بدل ما نعمل mask بعد الـ CAM (اللي ما يحل المشكلة)،
نربط الـ Loss نفسه بمواقع الـ Bounding Boxes في الـ prediction space.

الخطوات:
  1. نشغّل inference عادي للحصول على الـ Bounding Boxes
  2. نحوّل الـ boxes لـ 640×640 coordinate space
  3. نبني Spatial Weight Map — كل prediction قريبة من box تاخذ وزن عالي
  4. Loss = weighted sum of confidence scores (فقط predictions جوا الـ boxes)
  5. Gradients تشير للـ features اللي سببت هالـ detections تحديداً
  6. نطبق Gaussian Mask على الـ CAM النهائي كـ cleanup إضافي
"""

import cv2
import numpy as np
import torch

# ── Layer guide for YOLOv8 ───────────────────────────────────────────
#  layer[6]  → C2f  40×40  (best spatial resolution for aircraft bodies)
#  layer[9]  → SPPF 20×20  (strong semantics, good fallback)
#  layer[8]  → C2f  20×20  (second fallback)
# ────────────────────────────────────────────────────────────────────
CANDIDATE_LAYERS = [6, 9, 8]   # layer[6]=C2f 40×40 first (more spatial), fallback to SPPF(9)

# YOLOv8 anchor grid helper — maps flat index → (cx, cy) in 640 space
# YOLOv8 uses 3 strides: 8, 16, 32 → grids 80×80, 40×40, 20×20
# Total anchors = 6400 + 1600 + 400 = 8400
def _build_anchor_grid(device="cpu"):
    """Returns (8400, 2) tensor of (cx, cy) in 640×640 pixel space."""
    strides = [8, 16, 32]
    grids = []
    for s in strides:
        n = 640 // s
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        cx = (xs.flatten().float() + 0.5) * s
        cy = (ys.flatten().float() + 0.5) * s
        grids.append(torch.stack([cx, cy], dim=1))
    return torch.cat(grids, dim=0).to(device)  # (8400, 2)


class GradCAM:
    def __init__(self, yolov8_detector_wrapper):
        """
        yolov8_detector_wrapper : YOLOv8Detector custom wrapper
        """
        self.ultralytics_model = yolov8_detector_wrapper.model   # Ultralytics YOLO
        self.torch_model       = self.ultralytics_model.model     # PyTorch DetectionModel

        n_layers = len(self.torch_model.model)
        self.candidates = [i for i in CANDIDATE_LAYERS if i < n_layers]
        if not self.candidates:
            raise RuntimeError("[Grad-CAM] No valid candidate layers found.")

        # Pre-build anchor grid (stays constant per image size 640)
        self._anchor_grid = _build_anchor_grid()  # (8400, 2)

    # ─── Public API ────────────────────────────────────────────
    def generate_heatmap(self, image_np_rgb: np.ndarray) -> np.ndarray:
        """
        image_np_rgb : H×W×3 RGB numpy array
        Returns      : same shape RGB with heatmap blended in
        """
        h, w = image_np_rgb.shape[:2]

        # ── 1. Normal inference → get boxes in original image space ──
        boxes_orig = self._get_boxes(image_np_rgb, h, w)

        # ── 2. Convert boxes to 640×640 space for spatial loss ────────
        boxes_640 = []
        for (x1, y1, x2, y2) in boxes_orig:
            bx1 = int(x1 * 640 / w)
            by1 = int(y1 * 640 / h)
            bx2 = int(x2 * 640 / w)
            by2 = int(y2 * 640 / h)
            boxes_640.append((bx1, by1, bx2, by2))

        # ── 3. Generate raw CAM float map using spatial loss ──────────
        cam_float = None
        for layer_idx in self.candidates:
            cam_float = self._get_cam(image_np_rgb, layer_idx, boxes_640)
            if cam_float is not None:
                print(f"[Grad-CAM] Success with layer[{layer_idx}]")
                break

        if cam_float is None:
            print("[Grad-CAM] All layers gave empty CAM. Returning original.")
            return image_np_rgb

        # ── 4. Post-process: Gaussian mask + CLAHE + colorize + blend ─
        if boxes_orig:
            cam_float = self._apply_box_mask(cam_float, boxes_orig, h, w)

        cam_float   = self._sharpen_cam(cam_float)
        heatmap_u8  = np.uint8(255 * cam_float)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(image_np_rgb, 0.5, heatmap_rgb, 0.5, 0)

        return overlay

    def remove_hooks(self):
        """No-op — uses temporary forward hooks + retain_grad."""
        pass

    # ─── Step 1: Normal YOLO inference → boxes ─────────────────
    def _get_boxes(self, image_np_rgb, h, w, conf=0.20, pad=10):
        """
        Supports both horizontal boxes and OBB (Oriented Bounding Boxes).
        """
        boxes = []
        try:
            results = self.ultralytics_model.predict(
                image_np_rgb, conf=conf, verbose=False
            )
            if results and len(results) > 0:
                res = results[0]
                
                # Check for OBB (Oriented Bounding Boxes)
                if hasattr(res, 'obb') and res.obb is not None and len(res.obb) > 0:
                    # OBB.xyxy gives the horizontal bounding box for the oriented one
                    coords = res.obb.xyxy.cpu().numpy()
                    for box in coords:
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w, x2+pad), min(h, y2+pad)
                        boxes.append((x1, y1, x2, y2))
                
                # Fallback to standard boxes
                elif hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                    coords = res.boxes.xyxy.cpu().numpy()
                    for box in coords:
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w, x2+pad), min(h, y2+pad)
                        boxes.append((x1, y1, x2, y2))

        except Exception as e:
            print(f"[Grad-CAM] Box extraction failed: {e}")
        return boxes

    # ─── Step 2: Generate CAM with spatial loss ─────────────────
    def _get_cam(self, image_np_rgb: np.ndarray, layer_idx: int,
                 boxes_640: list) -> np.ndarray | None:
        """
        boxes_640 : boxes in 640×640 coordinate space
        Returns   : cam_float (H, W) in [0,1] or None
        """
        h, w = image_np_rgb.shape[:2]

        img_resized = cv2.resize(image_np_rgb, (640, 640))
        img_tensor  = (
            torch.from_numpy(img_resized.copy())
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )

        captured = {}

        def _fwd_hook(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            feat.retain_grad()
            captured["feat"] = feat

        handle = self.torch_model.model[layer_idx].register_forward_hook(_fwd_hook)

        self.torch_model.eval()
        try:
            with torch.inference_mode(mode=False):
                with torch.enable_grad():
                    img_g   = img_tensor.detach().clone().requires_grad_(True)
                    raw_out = self.torch_model(img_g)

                    # ── KEY FIX: spatial loss tied to box locations ──
                    loss = self._spatial_confidence_loss(raw_out, boxes_640)
                    if loss is None:
                        return None

                    self.torch_model.zero_grad()
                    loss.backward()
        finally:
            handle.remove()

        feat = captured.get("feat")
        if feat is None or feat.grad is None:
            print(f"[Grad-CAM] layer[{layer_idx}]: no gradient captured.")
            return None

        # ── Grad-CAM++ weights ────────────────────────────────────
        acts      = feat.detach()                    # (1, C, H_f, W_f)
        grads     = feat.grad.detach()               # (1, C, H_f, W_f)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom     = 2.0 * grads_sq + (acts * grads_cu).mean(dim=[2, 3], keepdim=True)
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / (denom + 1e-7)
        weights   = (alpha * torch.relu(grads)).mean(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze(0)  # (H_f, W_f)
        cam = torch.relu(cam).cpu().numpy()

        cam_max = cam.max()
        if cam_max < 1e-8:
            print(f"[Grad-CAM] layer[{layer_idx}]: CAM is empty.")
            return None

        cam = cam / cam_max
        cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        return cam_up  # (H, W) float [0,1]

    # ─── CORE FIX: Spatial confidence loss ─────────────────────
    def _spatial_confidence_loss(self, raw_out, boxes_640: list):
        """
        بدل ما ناخذ top-k predictions عشوائياً،
        نبني spatial weight لكل prediction بناءً على موقعها:
          - predictions جوا أو قريبة من الـ boxes → weight = 1.0
          - predictions بعيدة → weight ≈ 0.0

        Loss = sum(confidence[i] * spatial_weight[i])

        هذا يجبر الـ gradients تشير للـ features اللي سببت
        الـ detections في مواقع الطائرات تحديداً.
        """
        tensor = None
        if isinstance(raw_out, torch.Tensor):
            tensor = raw_out
        elif isinstance(raw_out, (list, tuple)):
            for x in raw_out:
                if isinstance(x, torch.Tensor) and x.ndim == 3:
                    tensor = x
                    break

        if tensor is None:
            return None

        device = tensor.device

        # cls scores: (1, C, 8400) → (8400,)
        cls_scores    = tensor[:, 4:, :].sigmoid()       # (1, C, 8400)
        max_scores, _ = cls_scores.max(dim=1)             # (1, 8400)
        max_scores    = max_scores.squeeze(0)             # (8400,)

        # ── Build spatial weight from boxes ──────────────────────
        if boxes_640:
            anchor_grid = self._anchor_grid.to(device)    # (8400, 2)
            spatial_weight = torch.zeros(8400, device=device)

            for (x1, y1, x2, y2) in boxes_640:
                cx1, cy1 = float(x1), float(y1)
                cx2, cy2 = float(x2), float(y2)

                # للـ predictions اللي مركزها جوا الـ box → weight 1.0
                inside_x = (anchor_grid[:, 0] >= cx1) & (anchor_grid[:, 0] <= cx2)
                inside_y = (anchor_grid[:, 1] >= cy1) & (anchor_grid[:, 1] <= cy2)
                inside   = inside_x & inside_y
                spatial_weight = torch.where(inside, torch.ones_like(spatial_weight), spatial_weight)

                # للـ predictions القريبة من حواف الـ box → weight 0.5 (soft margin)
                margin = 30.0  # pixels in 640 space
                near_x = (anchor_grid[:, 0] >= cx1 - margin) & (anchor_grid[:, 0] <= cx2 + margin)
                near_y = (anchor_grid[:, 1] >= cy1 - margin) & (anchor_grid[:, 1] <= cy2 + margin)
                near   = near_x & near_y & (~inside)
                spatial_weight = torch.where(near, torch.full_like(spatial_weight, 0.5), spatial_weight)

            # فقط الـ predictions اللي فوق threshold وجوا/قريب الـ boxes
            conf_mask     = max_scores > 0.15
            weighted_mask = spatial_weight > 0.0
            final_mask    = conf_mask & weighted_mask

            selected = max_scores[final_mask] * spatial_weight[final_mask]

            if selected.numel() > 0:
                return selected.sum()

        # Fallback: لو ما في boxes أو ما في predictions، top-20 فقط
        k = min(20, max_scores.numel())
        return torch.topk(max_scores.flatten(), k).values.sum()

    # ─── Post-process: Bounding Box Mask ───────────────────────
    def _apply_box_mask(self, cam_float, boxes_orig, h, w):
        """
        Hard mask — كل شي خارج الـ boxes = 0 تماماً
        """
        if not boxes_orig:
            return cam_float

        mask = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2) in boxes_orig:
            mask[y1:y2, x1:x2] = 1.0

        # Hard cut — لا Gaussian، لا soft edges
        cam_masked = cam_float * mask

        cam_max = cam_masked.max()
        if cam_max > 1e-8:
            cam_masked = cam_masked / cam_max

        return cam_masked

    # ─── CLAHE sharpening ──────────────────────────────────────
    @staticmethod
    def _sharpen_cam(cam_float: np.ndarray) -> np.ndarray:
        cam_u8  = np.uint8(255 * cam_float)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cam_eq  = clahe.apply(cam_u8)
        return cam_eq.astype(np.float32) / 255.0