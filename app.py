import json
import cv2
import numpy as np
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import (LEFT, BOTH, YES, DISABLED, NORMAL, END, TOP, X, Y, BOTTOM, RIGHT,
                                    HORIZONTAL, VERTICAL, NW) # Adicionado NW
from tkinter import (Canvas, filedialog, messagebox, simpledialog, Toplevel, Label, StringVar,
                     PhotoImage as tkPhotoImage, Text, scrolledtext) # Adicionado Text, scrolledtext
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from datetime import datetime
import sys
import os

# ---------- parâmetros globais ------------------------------------------------
MODEL_DIR = Path("modelos")
MODEL_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR = MODEL_DIR / "_templates"
TEMPLATE_DIR.mkdir(exist_ok=True)

# Limiares de inspeção
THR_CORR = 0.1  # Limiar para template matching (clips)
MIN_PX = 10      # Contagem mínima de pixels da cor para passar (clips e fitas)

# Parâmetros do Canvas e Preview
PREVIEW_W = 800  # Largura máxima do canvas para exibição inicial
PREVIEW_H = 600  # Altura máxima do canvas para exibição inicial

# Parâmetros ORB para registro de imagem
ORB_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Cores para desenho no canvas (Editor)
COLOR_CLIP = "cyan"
COLOR_FITA = "lime green"
COLOR_SELECTED = "red"
COLOR_DRAWING = "yellow"

# Cores para desenho no canvas (Inspeção)
COLOR_PASS = "green"
COLOR_FAIL = "red"
COLOR_ALIGN_FAIL = "orange" # Cor para slots se o alinhamento falhar


# ---------- utilidades --------------------------------------------------------
def cv2_to_tk(img_bgr, max_w=None, max_h=None):
    """
    Converte imagem OpenCV BGR para formato Tkinter PhotoImage,
    redimensionando para caber em max_w x max_h, se fornecido.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, 1.0
    h, w = img_bgr.shape[:2]
    scale = 1.0

    if max_w and w > max_w:
        scale = max_w / w
    if max_h and h * scale > max_h: # Verifica altura após escalar pela largura
        scale = max_h / h

    # Recalcula escala se altura for limitante primário
    if max_h and h > max_h and scale == 1.0: # Se só altura estourar
        scale = max_h / h
    if max_w and w * scale > max_w: # Verifica largura após escalar pela altura
        scale = max_w / w


    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Garante que as dimensões não sejam zero
        if new_w < 1: new_w = 1
        if new_h < 1: new_h = 1
        try:
            img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
             print(f"Erro ao redimensionar imagem: {e}. Dimensões: ({new_w}x{new_h})")
             return None, 1.0 # Retorna None em caso de erro no resize
    else:
        img_bgr_resized = img_bgr

    try:
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        photo_image = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        return photo_image, scale # Retorna a escala aplicada
    except Exception as e:
        print(f"Erro ao converter imagem para Tkinter: {e}")
        return None, scale


def cria_mascara_pixels(img, bgr, ht, st, vt):
    """Cria máscara de cor e retorna a máscara e a contagem de pixels não-zero."""
    if img is None or img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0:
        return np.zeros((1, 1), dtype=np.uint8), 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    try:
        # Usa np.uint8([[bgr]]) para garantir o formato correto para cvtColor
        h_base, s_base, v_base = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(int)
    except cv2.error as e:
        print(f"Erro ao converter cor BGR para HSV: {bgr} - {e}")
        return np.zeros(img.shape[:2], dtype=np.uint8), 0 # Retorna máscara vazia se a cor for inválida

    # Calcula limites inferior e superior
    lo = np.array([max(h_base - ht, 0), max(s_base - st, 0), max(v_base - vt, 0)])
    hi = np.array([min(h_base + ht, 179), min(s_base + st, 255), min(v_base + vt, 255)])

    # Tratamento especial para Hue (H) quando o intervalo cruza o 0 (ex: vermelho)
    if lo[0] > hi[0]: # O intervalo 'dá a volta' em 180/0
        # Cria duas máscaras: uma de lo[0] até 179, outra de 0 até hi[0]
        mask1 = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]]), np.array([179, hi[1], hi[2]]))
        mask2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), np.array([hi[0], hi[1], hi[2]]))
        mask = cv2.bitwise_or(mask1, mask2) # Combina as duas máscaras
    else:
        # Intervalo normal de H
        mask = cv2.inRange(hsv, lo, hi)

    pixels_count = cv2.countNonZero(mask)
    return mask, pixels_count


def pick_color(parent_window, roi):
    """Flow interativo para selecionar cor e ajustar limiares HSV."""
    if roi is None or roi.size == 0 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
        messagebox.showwarning("Aviso", "ROI vazia para seleção de cor.", parent=parent_window)
        return None

    sel = []
    window_name_pick = "PickColor - Clique na cor desejada (ESC para cancelar)"
    # Tenta tornar a janela 'topmost' para facilitar a seleção
    cv2.namedWindow(window_name_pick, cv2.WND_PROP_TOPMOST)
    cv2.setWindowProperty(window_name_pick, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(window_name_pick,
                         lambda e, x, y, f, p: sel.extend([x, y]) if e == cv2.EVENT_LBUTTONDOWN else None)

    print("Esperando clique para seleção de cor...")
    while not sel:
        # Exibe ROI um pouco maior para facilitar clique nas bordas
        display_roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[128,128,128])
        cv2.imshow(window_name_pick, display_roi)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            cv2.destroyWindow(window_name_pick)
            print("Seleção de cor cancelada (Esc).")
            return None
        try:
            if cv2.getWindowProperty(window_name_pick, cv2.WND_PROP_VISIBLE) < 1:
                print("Seleção de cor cancelada (janela fechada).")
                return None
        except cv2.error:
            print("Seleção de cor cancelada (erro ao verificar janela).")
            return None

    cv2.destroyWindow(window_name_pick)
    # Ajusta coordenadas do clique por causa da borda adicionada
    x_sel, y_sel = sel[0] - 5, sel[1] - 5
    # Garante que as coordenadas estejam dentro dos limites da ROI *original*
    y_sel = max(0, min(y_sel, roi.shape[0] - 1))
    x_sel = max(0, min(x_sel, roi.shape[1] - 1))
    bgr = roi[y_sel, x_sel].tolist()
    print(f"Cor BGR selecionada: {bgr} em ({x_sel}, {y_sel})")


    window_name_adjust = "MaskAdjust - Ajuste os sliders (Enter=OK, Esc=Cancelar)"
    cv2.namedWindow(window_name_adjust, cv2.WND_PROP_TOPMOST)
    cv2.setWindowProperty(window_name_adjust, cv2.WND_PROP_TOPMOST, 1)
    try:
        cv2.resizeWindow(window_name_adjust, max(roi.shape[1]*3, 400), max(roi.shape[0] + 100, 250)) # Ajusta tamanho
    except cv2.error:
        pass

    initial_h = 10
    initial_s = 60
    initial_v = 60

    cv2.createTrackbar("H +/-", window_name_adjust, initial_h, 90, lambda *_: None) # Tolerância H
    cv2.createTrackbar("S +/-", window_name_adjust, initial_s, 128, lambda *_: None) # Tolerância S
    cv2.createTrackbar("V +/-", window_name_adjust, initial_v, 128, lambda *_: None) # Tolerância V

    ht, st, vt = initial_h, initial_s, initial_v

    print("Esperando ajuste de máscara (Enter/Esc)...")
    while True:
        try:
             if cv2.getWindowProperty(window_name_adjust, cv2.WND_PROP_VISIBLE) < 1:
                   print("Ajuste de máscara cancelado (janela fechada).")
                   return None
        except cv2.error:
             print("Ajuste de máscara cancelado (erro ao verificar janela).")
             return None

        ht = cv2.getTrackbarPos("H +/-", window_name_adjust)
        st = cv2.getTrackbarPos("S +/-", window_name_adjust)
        vt = cv2.getTrackbarPos("V +/-", window_name_adjust)

        mask, pixels_count = cria_mascara_pixels(roi, bgr, ht, st, vt)

        # Cria a imagem combinada para visualização
        if mask is not None and mask.size > 0:
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display_masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

            target_h, target_w = roi.shape[:2]
            display_mask_resized = cv2.resize(display_mask, (target_w, target_h))
            display_masked_roi_resized = cv2.resize(display_masked_roi, (target_w, target_h))

            if roi is not None and roi.size > 0 and display_mask_resized is not None and display_mask_resized.size > 0 and display_masked_roi_resized is not None and display_masked_roi_resized.size > 0:
                 try:
                      # Adiciona texto com contagem de pixels
                      roi_with_text = roi.copy()
                      cv2.putText(roi_with_text, f"Pixels: {pixels_count}", (5, target_h - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA) # Contorno preto
                      cv2.putText(roi_with_text, f"Pixels: {pixels_count}", (5, target_h - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Texto branco

                      comb = np.hstack([roi_with_text, display_mask_resized, display_masked_roi_resized])
                      cv2.imshow(window_name_adjust, comb)
                 except Exception as e_stack:
                      print(f"Erro ao empilhar imagens para visualização: {e_stack}")
                      cv2.imshow(window_name_adjust, roi) # Mostra apenas a ROI em caso de erro
            else:
                 cv2.imshow(window_name_adjust, roi) # Mostra apenas a ROI
        else:
             cv2.imshow(window_name_adjust, roi) # Mostra apenas a ROI


        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # Enter
            cv2.destroyWindow(window_name_adjust)
            print(f"Máscara salva. Cor BGR: {bgr}, Limiares HSV: H={ht}, S={st}, V={vt}, Pixels: {pixels_count}")
            return bgr, (ht, st, vt)
        elif key == 27: # Esc
            cv2.destroyWindow(window_name_adjust)
            print("Ajuste de máscara cancelado (Esc).")
            return None

    try:
        cv2.destroyWindow(window_name_adjust)
    except cv2.error:
        pass
    return None


# ---------- Registro de Imagem (ORB) - Sem alterações ---------------------------
try:
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_N_LEVELS)
    print("Detector ORB inicializado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar ORB: {e}. O registro de imagem não funcionará.")
    orb = None

def find_image_transform(img_ref, img_test):
    if orb is None:
        print("ORB não inicializado. Pulando find_image_transform.")
        return None, None, "ORB não inicializado." # Retorna erro
    if img_ref is None or img_test is None or img_ref.size == 0 or img_test.size == 0:
         return None, None, "Imagem de referência ou teste vazia." # Retorna erro

    try:
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        img_test_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

        kp_ref, des_ref = orb.detectAndCompute(img_ref_gray, None)
        kp_test, des_test = orb.detectAndCompute(img_test_gray, None)

        if des_ref is None or des_test is None or len(des_ref) < 4 or len(des_test) < 4:
            err_msg = f"Descritores insuficientes: Ref ({len(des_ref) if des_ref is not None else 0}), Teste ({len(des_test) if des_test is not None else 0})."
            return None, None, err_msg

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref, des_test)
        matches = sorted(matches, key=lambda x: x.distance)
        MIN_MATCH_COUNT = 10

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.99, maxIters=1000)

            if M is None:
                 return None, None, "findHomography falhou (M is None)."
            inliers = np.sum(mask) if mask is not None else 0
            if inliers < MIN_MATCH_COUNT / 2:
                 return None, None, f"findHomography encontrou poucos inliers ({inliers})."

            # Opcional: Visualizar matches para debug
            # img_matches = cv2.drawMatches(img_ref_gray, kp_ref, img_test_gray, kp_test, matches[:50], None, matchColor=(0,255,0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
            # cv2.imshow("ORB Matches (Inliers)", img_matches)
            # cv2.waitKey(1)

            return M, mask, None # Sucesso, retorna M, mask, e None para erro
        else:
            return None, None, f"Matches insuficientes ({len(matches)} < {MIN_MATCH_COUNT})."
    except cv2.error as e:
        return None, None, f"Erro no OpenCV durante alinhamento: {e}"
    except Exception as e:
        return None, None, f"Erro inesperado durante alinhamento: {e}"


def transform_rectangle(rect, M, img_shape):
    if M is None:
        x, y, w, h = rect
        # Retorna cantos no formato (4, 2) e o bbox original
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return rect, corners.astype(int)

    x, y, w, h = rect
    # Cantos como (N, 1, 2) para perspectiveTransform
    corners_orig = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)

    try:
        transformed_corners_float = cv2.perspectiveTransform(corners_orig, M)
        if transformed_corners_float is None:
             print("Aviso: perspectiveTransform retornou None.")
             return [0, 0, 0, 0], None

        # Converte cantos para inteiros (formato (4, 2))
        transformed_corners_int = transformed_corners_float.reshape(-1, 2).astype(int)

        # Calcula o bounding box dos cantos transformados
        x_coords = transformed_corners_int[:, 0]
        y_coords = transformed_corners_int[:, 1]

        x_min = np.min(x_coords)
        y_min = np.min(y_coords)
        x_max = np.max(x_coords)
        y_max = np.max(y_coords)

        h_img, w_img = img_shape[:2]
        x_min_int = max(0, x_min) # Não usa floor/ceil aqui, usa min/max diretos dos cantos int
        y_min_int = max(0, y_min)
        # Limita o max ao tamanho da imagem - 1 (índice máximo)
        x_max_int = min(w_img - 1, x_max)
        y_max_int = min(h_img - 1, y_max)

        # Calcula largura e altura, garantindo >= 0
        w_new = max(0, x_max_int - x_min_int + 1) # +1 para incluir o pixel max
        h_new = max(0, y_max_int - y_min_int + 1)

        transformed_rect_bbox = [x_min_int, y_min_int, w_new, h_new]

        # Retorna o bbox e os 4 cantos transformados
        return transformed_rect_bbox, transformed_corners_int

    except cv2.error as e:
        print(f"Erro em perspectiveTransform: {e}")
        return [0, 0, 0, 0], None
    except Exception as e:
        print(f"Erro inesperado em transform_rectangle: {e}")
        return [0, 0, 0, 0], None


# ---------- Verificação do Slot - Sem alterações funcionais, mas com logging -------------

def check_slot(img_test, slot_data, M):
    """Verifica um slot na imagem de teste usando a transformação M."""
    slot_id = slot_data.get('id', 'N/A')
    ref_rect = slot_data["rect"]
    slot_type = slot_data["type"]
    bgr_color = slot_data["color"]
    hsv_thresholds = (slot_data["h"], slot_data["s"], slot_data["v"])
    log_msgs = [] # Log específico para este slot

    transformed_rect_bbox, transformed_corners = transform_rectangle(ref_rect, M, img_test.shape)
    tx, ty, tw, th = transformed_rect_bbox

    correlation = 0.0
    pixels_count = 0
    template_ok = True # Assume OK se não for clip
    color_ok = True

    # Verifica validade do retângulo transformado
    if tw <= 0 or th <= 0 or ty < 0 or tx < 0 or ty + th > img_test.shape[0] or tx + tw > img_test.shape[1]:
        log_msgs.append(f"Fora dos limites ou inválido após transformação: BBox={transformed_rect_bbox}")
        return False, 0.0, 0, transformed_corners, transformed_rect_bbox, log_msgs

    # Extrai ROI
    roi_test = img_test[ty:ty + th, tx:tx + tw]
    if roi_test is None or roi_test.size == 0 or roi_test.shape[0] <= 0 or roi_test.shape[1] <= 0:
        log_msgs.append(f"ROI extraída vazia em BBox={transformed_rect_bbox}")
        return False, 0.0, 0, transformed_corners, transformed_rect_bbox, log_msgs

    # 1. Verificação de Cor
    try:
        mask, pixels_count = cria_mascara_pixels(roi_test, bgr_color, *hsv_thresholds)
        color_ok = pixels_count >= MIN_PX
        log_msgs.append(f"Cor: Pixels={pixels_count} (Min={MIN_PX}) -> {'OK' if color_ok else 'FALHA'}")
    except Exception as e:
        log_msgs.append(f"Erro na verificação de cor: {e}")
        color_ok = False

    # 2. Verificação de Template (se for clip)
    if slot_type == "clip":
        template_path_str = slot_data.get("template") # Deve ser o nome do arquivo
        template_path = None
        if template_path_str:
             # Procura primeiro no diretório global de templates
             candidate_path = TEMPLATE_DIR / template_path_str
             if candidate_path.exists():
                  template_path = candidate_path
             else:
                  # Poderia tentar outros locais, mas vamos manter simples por agora
                  log_msgs.append(f"Template '{template_path_str}' não encontrado em {TEMPLATE_DIR}")
                  template_ok = False

        if template_ok and (not template_path or not template_path.exists()):
             log_msgs.append(f"Template não encontrado: {template_path_str}")
             template_ok = False
        elif template_ok: # Se o path existe e ainda estamos OK
             template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
             if template is None or template.size == 0:
                  log_msgs.append(f"Template vazio ou não carregado: {template_path}")
                  template_ok = False
             else:
                  # Redimensionamento e Template Matching
                  hR, wR = roi_test.shape[:2]
                  hT, wT = template.shape[:2]

                  if hR > 0 and wR > 0 and hT > 0 and wT > 0:
                       scale_factor = min(hR / hT, wR / wT) if hT > 0 and wT > 0 else 1.0
                       new_w = max(1, int(wT * scale_factor))
                       new_h = max(1, int(hT * scale_factor))

                       template_resized = cv2.resize(template, (new_w, new_h))
                       hT_res, wT_res = template_resized.shape[:2]

                       if hT_res <= hR and wT_res <= wR and hT_res > 0 and wT_res > 0:
                            try:
                                 roi_test_gray = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY)
                                 result = cv2.matchTemplate(roi_test_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                                 _, max_val, _, _ = cv2.minMaxLoc(result)
                                 correlation = max_val
                                 template_ok = correlation >= THR_CORR
                                 log_msgs.append(f"Template: Corr={correlation:.3f} (Lim={THR_CORR}) -> {'OK' if template_ok else 'FALHA'}")
                            except cv2.error as tm_error:
                                 log_msgs.append(f"Erro no matchTemplate: {tm_error}")
                                 template_ok = False
                                 correlation = -1.0 # Indica erro
                       else:
                            log_msgs.append(f"Template redimensionado ({wT_res}x{hT_res}) não cabe na ROI ({wR}x{hR})")
                            template_ok = False
                  else:
                       log_msgs.append("ROI ou Template com dimensão zero para matching")
                       template_ok = False

    # Resultado final do slot
    if slot_type == "clip":
        is_ok = color_ok and template_ok
    elif slot_type == "fita":
        is_ok = color_ok
    else:
        is_ok = False
        log_msgs.append(f"Tipo de slot desconhecido: {slot_type}")

    log_msgs.insert(0, f"Resultado Slot {slot_id} ({slot_type}): {'PASS' if is_ok else 'FAIL'}") # Adiciona resultado geral no início
    return is_ok, correlation, pixels_count, transformed_corners, transformed_rect_bbox, log_msgs


# ---------- Diálogo de Edição de Slot (sem alterações) -----------------------
class EditSlotDialog(Toplevel):
    def __init__(self, parent, slot_data, malha_frame_instance):
        super().__init__(parent)
        self.parent = parent # Janela principal
        self.malha_frame = malha_frame_instance # Instância da MalhaFrame
        self.slot_data_original = slot_data.copy() # Guarda original para cancelamento
        self.slot_data_edited = slot_data.copy() # Trabalha neste

        self.title(f"Editar Slot {self.slot_data_edited['id']}")
        self.transient(parent)
        self.grab_set() # Impede interação com a janela principal
        self.resizable(False, False)
        self.attributes('-topmost', True) # Tenta manter no topo

        # --- Widgets ---
        frame = ttk.Frame(self, padding=10)
        frame.pack(expand=True, fill=BOTH)

        # ID (Read-only)
        ttk.Label(frame, text="ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.id_label = ttk.Label(frame, text=str(self.slot_data_edited['id']))
        self.id_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Tipo (Combobox)
        ttk.Label(frame, text="Tipo:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.type_var = StringVar(value=self.slot_data_edited['type'])
        self.type_combo = Combobox(frame, textvariable=self.type_var, values=["clip", "fita"], state="readonly", width=8)
        self.type_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.type_combo.bind("<<ComboboxSelected>>", self.update_template_visibility) # Atualiza visibilidade do template

        # Cor
        ttk.Label(frame, text="Cor:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.color_canvas = Canvas(frame, width=30, height=20, bg=self.get_hex_color(self.slot_data_edited['color']))
        self.color_canvas.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.pick_color_button = ttk.Button(frame, text="Selecionar Cor...", command=self.pick_new_color, width=15)
        self.pick_color_button.grid(row=2, column=2, padx=5, pady=5)

        # Limiares HSV (Read-only por enquanto)
        ttk.Label(frame, text="Limiares (H, S, V):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.hsv_label = ttk.Label(frame, text=f"({self.slot_data_edited['h']}, {self.slot_data_edited['s']}, {self.slot_data_edited['v']})")
        self.hsv_label.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        # Template Path (mostra/oculta baseado no tipo)
        self.template_label_widget = ttk.Label(frame, text="Template:")
        self.template_path_label = ttk.Label(frame, text=self.slot_data_edited.get('template', 'N/A'), wraplength=250) # wraplength para caminhos longos

        # --- Botões de Ação ---
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=X, padx=10, pady=(5, 10)) # Adiciona um pouco de pady top

        self.save_button = ttk.Button(button_frame, text="Salvar Alterações", command=self.save_changes, bootstyle="success")
        self.save_button.pack(side=RIGHT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancelar", command=self.cancel, bootstyle="secondary")
        self.cancel_button.pack(side=RIGHT, padx=5)

        self.update_template_visibility() # Chama para definir estado inicial

        # Posiciona a janela no centro da janela pai
        parent.update_idletasks() # Garante que as dimensões do pai estão atualizadas
        x_parent = parent.winfo_rootx()
        y_parent = parent.winfo_rooty()
        w_parent = parent.winfo_width()
        h_parent = parent.winfo_height()
        w_dialog = self.winfo_reqwidth() # Largura requisitada pelo dialog
        h_dialog = self.winfo_reqheight() # Altura requisitada pelo dialog
        x_dialog = x_parent + (w_parent // 2) - (w_dialog // 2)
        y_dialog = y_parent + (h_parent // 2) - (h_dialog // 2)
        self.geometry(f"+{x_dialog}+{y_dialog}")


        self.protocol("WM_DELETE_WINDOW", self.cancel) # Lidar com fechamento pelo 'X'
        self.wait_window() # Espera até que a janela seja fechada

    def get_hex_color(self, bgr):
        """Converte cor BGR (lista ou tuple) para string hexadecimal #RRGGBB."""
        if isinstance(bgr, (list, tuple)) and len(bgr) == 3:
             # A cor BGR é [B, G, R], Hex é #RRGGBB
             # Garante que os valores estejam dentro de 0-255
             b = max(0, min(255, int(bgr[0])))
             g = max(0, min(255, int(bgr[1])))
             r = max(0, min(255, int(bgr[2])))
             return f"#{r:02x}{g:02x}{b:02x}"
        return "#FFFFFF" # Branco como fallback

    def update_template_visibility(self, event=None):
        """Mostra ou oculta os widgets de template baseado no tipo selecionado."""
        if self.type_var.get() == "clip":
            self.template_label_widget.grid(row=4, column=0, padx=5, pady=(10,5), sticky="nw") # Adiciona pady top
            self.template_path_label.grid(row=4, column=1, columnspan=2, padx=5, pady=(10,5), sticky="w")
            self.template_path_label.config(text=self.slot_data_edited.get('template', 'N/A') or 'N/A') # Garante que mostre N/A se for None
        else:
            self.template_label_widget.grid_remove()
            self.template_path_label.grid_remove()

    def pick_new_color(self):
        """Pega a ROI atual e chama pick_color para atualizar a cor e limiares."""
        if not self.malha_frame.img_cv or self.malha_frame.img_cv.size == 0:
            messagebox.showerror("Erro", "Imagem de referência não está carregada.", parent=self)
            return

        xr, yr, wr, hr = self.slot_data_edited['rect']
        img_h, img_w = self.malha_frame.img_cv.shape[:2]
        y_start = max(0, yr); y_end = min(img_h, yr + hr)
        x_start = max(0, xr); x_end = min(img_w, xr + wr)

        if y_start >= y_end or x_start >= x_end:
             messagebox.showerror("Erro", "Região do Slot (ROI) inválida ou fora da imagem.", parent=self)
             return
        roi_original = self.malha_frame.img_cv[y_start:y_end, x_start:x_end]
        if roi_original is None or roi_original.size == 0:
             messagebox.showerror("Erro", "Não foi possível extrair a ROI para seleção de cor.", parent=self)
             return

        color_result = pick_color(self, roi_original) # Passa esta janela como pai
        if color_result:
            bgr_color, (ht, st, vt) = color_result
            self.slot_data_edited['color'] = bgr_color
            self.slot_data_edited['h'] = ht
            self.slot_data_edited['s'] = st
            self.slot_data_edited['v'] = vt
            self.color_canvas.config(bg=self.get_hex_color(bgr_color))
            self.hsv_label.config(text=f"({ht}, {st}, {vt})")
            print(f"Nova cor/limiares selecionados para slot {self.slot_data_edited['id']}")

    def save_changes(self):
        """Valida os dados e envia de volta para MalhaFrame."""
        new_type = self.type_var.get()
        self.slot_data_edited['type'] = new_type

        if new_type == 'fita':
             # Se mudou para 'fita', apenas garante que o template seja None
             if self.slot_data_edited.get('template'):
                  print(f"Tipo mudado para fita, definindo template como None (arquivo {self.slot_data_edited['template']} não excluído).")
                  self.slot_data_edited['template'] = None
        elif new_type == 'clip' and not self.slot_data_edited.get('template'):
             # Se mudou para 'clip' e não há template, informa o usuário
             messagebox.showinfo("Aviso", "Slot alterado para 'clip', mas nenhum template está associado. Crie um novo slot ou use uma função de redimensionamento (não implementada) para gerar o template.", parent=self)
             self.slot_data_edited['template'] = None

        self.malha_frame.update_slot_data(self.slot_data_edited)
        self.destroy()

    def cancel(self):
        """Cancela as alterações e fecha o diálogo."""
        print(f"Edição do slot {self.slot_data_original['id']} cancelada.")
        self.destroy()


# ---------- Aba Malha (Editor - Sem alterações funcionais) -------------------
class MalhaFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5)
        self.img_path = None
        self.img_cv = None
        self.tk_img = None
        self.scale = 1.0
        self.slots = []
        self.rect_drawing_id = None
        self.start_coords = None
        self.last_slot_id = 0
        self.selected_slot_id = None
        self.current_model_path = None

        # --- Frame de Botões Superior ---
        top_button_frame = ttk.Frame(self)
        top_button_frame.pack(side=TOP, fill=X, padx=2, pady=(2, 5))

        action_button_frame = ttk.LabelFrame(top_button_frame, text=" Arquivo ")
        action_button_frame.pack(side=LEFT, padx=5, fill=Y)
        ttk.Button(action_button_frame, text="Abrir Imagem", bootstyle="secondary",
                   command=self.browse_new_image).pack(side=TOP, padx=5, pady=2, fill=X)
        ttk.Button(action_button_frame, text="Carregar Modelo", bootstyle="info",
                   command=self.browse_load_model).pack(side=TOP, padx=5, pady=2, fill=X)
        self.save_button = ttk.Button(action_button_frame, text="Salvar Modelo", bootstyle="success",
                   command=self.save_model, state=DISABLED)
        self.save_button.pack(side=TOP, padx=5, pady=2, fill=X)

        edit_button_frame = ttk.LabelFrame(top_button_frame, text=" Slots ")
        edit_button_frame.pack(side=LEFT, padx=5, fill=Y)
        self.clear_button = ttk.Button(edit_button_frame, text="Limpar Slots", bootstyle="warning",
                   command=self.clear_slots, state=DISABLED)
        self.clear_button.pack(side=TOP, padx=5, pady=2, fill=X)
        self.edit_button = ttk.Button(edit_button_frame, text="Editar Selecionado", bootstyle="primary",
                   command=self.edit_selected_slot, state=DISABLED)
        self.edit_button.pack(side=TOP, padx=5, pady=2, fill=X)
        self.delete_button = ttk.Button(edit_button_frame, text="Excluir Selecionado", bootstyle="danger",
                   command=self.delete_selected_slot, state=DISABLED)
        self.delete_button.pack(side=TOP, padx=5, pady=2, fill=X)

        help_button_frame = ttk.LabelFrame(top_button_frame, text=" Ajuda ")
        help_button_frame.pack(side=LEFT, padx=5, fill=Y)
        ttk.Button(help_button_frame, text="Instruções", bootstyle="info-outline",
                   command=self.show_help).pack(side=TOP, padx=5, pady=2, fill=X)

        # --- Canvas com Barras de Rolagem ---
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=TOP, fill=BOTH, expand=YES, pady=5)

        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient="vertical")
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient="horizontal")
        self.v_scroll.pack(side=RIGHT, fill=Y)
        self.h_scroll.pack(side=BOTTOM, fill=X)

        # Define a cor de fundo do canvas baseada no tema (experimental)
        try:
            style = ttk.Style.get_instance()
            canvas_bg = style.lookup('TFrame', 'background') # Pega cor de fundo do frame
        except Exception:
            canvas_bg = "#333333" # Fallback para cor escura

        self.canvas = Canvas(self.canvas_frame, bg=canvas_bg,
                             yscrollcommand=self.v_scroll.set,
                             xscrollcommand=self.h_scroll.set,
                             highlightthickness=0)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=YES)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.update_button_states()

    def _clear_all(self):
        print("Limpando estado da MalhaFrame...")
        self.img_path = None
        self.img_cv = None
        self.tk_img = None # PhotoImage será liberado pelo GC
        self.scale = 1.0
        self.slots = []
        if self.rect_drawing_id:
            self.canvas.delete(self.rect_drawing_id)
            self.rect_drawing_id = None
        self.start_coords = None
        self.last_slot_id = 0
        self.selected_slot_id = None
        self.current_model_path = None
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, 0, 0))
        self.update_button_states()

    def _load_image_data(self, image_path):
        try:
            img_path_obj = Path(image_path)
            if not img_path_obj.exists():
                 messagebox.showerror("Erro", f"Arquivo de imagem não encontrado:\n{image_path}", parent=self)
                 return False

            self.img_cv = cv2.imread(str(img_path_obj))
            if self.img_cv is None or self.img_cv.size == 0:
                messagebox.showerror("Erro", f"Não foi possível carregar a imagem:\n{image_path}", parent=self)
                self.img_cv = None
                return False

            self.img_path = str(img_path_obj.resolve())
            self.canvas.delete("all")

            # Usa PREVIEW_W e PREVIEW_H para limitar o tamanho inicial
            self.tk_img, self.scale = cv2_to_tk(self.img_cv, PREVIEW_W, PREVIEW_H)

            if not self.tk_img:
                 messagebox.showerror("Erro", "Falha ao converter imagem para exibição.", parent=self)
                 self.img_cv = None; self.img_path = None
                 return False

            canvas_img_w = self.tk_img.width()
            canvas_img_h = self.tk_img.height()
            self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img, tags="background_image")
            self.canvas.config(scrollregion=(0, 0, canvas_img_w, canvas_img_h))

            print(f"Imagem carregada: {self.img_path}")
            print(f"Dims Orig: {self.img_cv.shape[1]}x{self.img_cv.shape[0]}, Canvas: {canvas_img_w}x{canvas_img_h}, Escala: {self.scale:.3f}")
            self.save_button.config(state=NORMAL)
            self.clear_button.config(state=NORMAL)
            return True
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro ao carregar a imagem:\n{e}", parent=self)
            self._clear_all()
            return False

    def browse_new_image(self):
        filepath = filedialog.askopenfilename(
            title="Abrir Imagem de Referência",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")]
        )
        if not filepath: return
        self._clear_all()
        if self._load_image_data(filepath):
             self.update_button_states()

    def browse_load_model(self):
        filepath = filedialog.askopenfilename(
            title="Carregar Modelo (.json)", initialdir=MODEL_DIR,
            filetypes=[("Modelo JSON", "*.json"), ("Todos", "*.*")]
        )
        if not filepath: return
        self.load_model(filepath)

    def load_model(self, model_path):
        try:
            model_path_obj = Path(model_path)
            with open(model_path_obj, 'r') as f:
                model_data = json.load(f)
            if "image_path" not in model_data or "slots" not in model_data:
                 messagebox.showerror("Erro", "Arquivo de modelo inválido.", parent=self); return
            self._clear_all()

            ref_image_path_str = model_data["image_path"]
            ref_image_path = Path(ref_image_path_str)
            if not ref_image_path.is_absolute(): # Se não for absoluto, tenta relativo ao JSON
                 candidate_path = model_path_obj.parent / ref_image_path
                 if candidate_path.exists():
                      ref_image_path = candidate_path
                 else: # Tenta relativo ao CWD como último recurso
                     if not ref_image_path.exists():
                         messagebox.showerror("Erro", f"Imagem de referência não encontrada:\n{ref_image_path_str}", parent=self)
                         self._clear_all(); return

            if not self._load_image_data(str(ref_image_path.resolve())):
                messagebox.showerror("Erro", f"Falha ao carregar imagem do modelo:\n{ref_image_path}", parent=self)
                return

            self.slots = model_data["slots"]
            max_id = 0
            valid_slots = []
            for i, slot in enumerate(self.slots):
                if not all(k in slot for k in ("id", "rect", "type", "color", "h", "s", "v")) or \
                   not isinstance(slot['rect'], list) or len(slot['rect']) != 4:
                    print(f"Aviso: Slot {slot.get('id', i)} inválido/incompleto no JSON, ignorado.")
                    continue
                if slot['type'] == 'clip' and 'template' not in slot: slot['template'] = None
                slot['canvas_id'] = None
                valid_slots.append(slot)
                if isinstance(slot.get('id'), int) and slot['id'] > max_id: max_id = slot['id']
            self.slots = valid_slots
            self.last_slot_id = max_id
            self.current_model_path = str(model_path_obj.resolve())
            print(f"Modelo '{model_path_obj.name}' carregado com {len(self.slots)} slots.")
            self.redraw_slots()
            self.update_button_states()
        except Exception as e:
            messagebox.showerror("Erro ao Carregar Modelo", f"Ocorreu um erro:\n{e}", parent=self)
            self._clear_all()

    def redraw_slots(self):
        self.canvas.delete("slot")
        self.selected_slot_id = None
        if self.img_cv is None or self.scale == 0:
            return


        for slot in self.slots:
            xr, yr, wr, hr = slot['rect']
            xa, ya = xr * self.scale, yr * self.scale
            wa, ha = wr * self.scale, hr * self.scale
            outline_color = COLOR_CLIP if slot['type'] == 'clip' else COLOR_FITA
            canvas_id = self.canvas.create_rectangle(
                xa, ya, xa + wa, ya + ha,
                outline=outline_color, width=2, tags=("slot", f"slot_{slot['id']}")
            )
            slot['canvas_id'] = canvas_id
        self.update_button_states()

    def on_canvas_press(self, event):
        """Chamado quando o botão 1 do mouse é pressionado no canvas."""
        # Verifica se a imagem existe e tem conteúdo válido
        if self.img_cv is None or self.img_cv.size == 0: return # Precisa de imagem válida

        # Converte coordenadas do evento para coordenadas do canvas (considerando scroll)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Verifica se o clique foi dentro de um slot existente
        clicked_slot_id = self.find_slot_at(canvas_x, canvas_y)

        if clicked_slot_id is not None:
            # --- Clique em um slot existente: Inicia seleção ---
            self.select_slot(clicked_slot_id)
            # Não inicia desenho (self.start_coords = None)
            self.start_coords = None
            if self.rect_drawing_id: # Remove retângulo de desenho se existir
                 self.canvas.delete(self.rect_drawing_id)
                 self.rect_drawing_id = None
        else:
            # --- Clique fora de um slot: Inicia desenho de novo slot ---
            self.deselect_all_slots() # Garante que nada esteja selecionado
            self.start_coords = (canvas_x, canvas_y)

            # Garante que o início esteja dentro dos limites da imagem no canvas
            # Verifica se tk_img existe antes de acessar width/height
            if self.tk_img:
                img_w_canvas = self.tk_img.width()
                img_h_canvas = self.tk_img.height()
                self.start_coords = (max(0, min(self.start_coords[0], img_w_canvas)),
                                     max(0, min(self.start_coords[1], img_h_canvas)))
            else:
                 # Se tk_img não existe (não deveria acontecer se img_cv existe, mas por segurança)
                 self.start_coords = (max(0, self.start_coords[0]), max(0, self.start_coords[1]))


            # Cria o retângulo temporário para desenho
            if self.rect_drawing_id:
                 self.canvas.delete(self.rect_drawing_id) # Deleta o antigo se houver
            self.rect_drawing_id = self.canvas.create_rectangle(
                self.start_coords[0], self.start_coords[1],
                self.start_coords[0], self.start_coords[1],
                outline=COLOR_DRAWING, width=2, tags="drawing_rect" # Tag específica
            )
            # print(f"Iniciando desenho em {self.start_coords}") # DEBUG

    def on_canvas_drag(self, event):
        if not self.rect_drawing_id or not self.start_coords: return
        current_x = self.canvas.canvasx(event.x); current_y = self.canvas.canvasy(event.y)
        img_w_canvas = self.tk_img.width(); img_h_canvas = self.tk_img.height()
        current_x = max(0, min(current_x, img_w_canvas)); current_y = max(0, min(current_y, img_h_canvas))
        self.canvas.coords(self.rect_drawing_id, self.start_coords[0], self.start_coords[1], current_x, current_y)

    def on_canvas_release(self, event):
        if not self.rect_drawing_id or not self.start_coords: return
        end_x = self.canvas.canvasx(event.x); end_y = self.canvas.canvasy(event.y)
        img_w_canvas = self.tk_img.width(); img_h_canvas = self.tk_img.height()
        end_x = max(0, min(end_x, img_w_canvas)); end_y = max(0, min(end_y, img_h_canvas))
        start_x, start_y = self.start_coords
        self.canvas.delete(self.rect_drawing_id); self.rect_drawing_id = None; self.start_coords = None
        xa1, xa2 = min(start_x, end_x), max(start_x, end_x)
        ya1, ya2 = min(start_y, end_y), max(start_y, end_y)
        wa, ha = xa2 - xa1, ya2 - ya1
        min_dim_canvas = 10
        if wa < min_dim_canvas or ha < min_dim_canvas:
            messagebox.showwarning("Aviso", f"Retângulo muito pequeno ({wa:.0f}x{ha:.0f} pixels). Mínimo {min_dim_canvas}x{min_dim_canvas}.", parent=self)
            return
        self.add_slot(xa1, ya1, wa, ha)

    def find_slot_at(self, canvas_x, canvas_y):
        overlapping_items = self.canvas.find_overlapping(canvas_x, canvas_y, canvas_x, canvas_y)
        for item_id in reversed(overlapping_items):
            tags = self.canvas.gettags(item_id)
            if "slot" in tags:
                for slot in self.slots:
                    if slot.get('canvas_id') == item_id: return slot['id']
        return None

    def select_slot(self, slot_id):
        if self.selected_slot_id == slot_id: return
        self.deselect_all_slots()
        target_slot = next((s for s in self.slots if s['id'] == slot_id), None)
        if target_slot and target_slot.get('canvas_id'):
            self.selected_slot_id = slot_id
            try:
                if self.canvas.find_withtag(f"slot_{slot_id}"): # Verifica se ainda existe
                     self.canvas.itemconfig(target_slot['canvas_id'], outline=COLOR_SELECTED, width=3)
                     print(f"Slot {slot_id} selecionado.")
                else:
                     print(f"Aviso: Item do canvas para slot {slot_id} não encontrado para seleção.")
                     self.selected_slot_id = None # Limpa se o item sumiu
            except Exception as e:
                 print(f"Erro ao selecionar slot {slot_id}: {e}")
                 self.selected_slot_id = None
        else:
             self.selected_slot_id = None
        self.update_button_states()

    def deselect_all_slots(self):
        if self.selected_slot_id is not None:
            slot = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
            if slot and slot.get('canvas_id'):
                original_color = COLOR_CLIP if slot['type'] == 'clip' else COLOR_FITA
                try:
                    if self.canvas.find_withtag(f"slot_{self.selected_slot_id}"):
                         self.canvas.itemconfig(slot['canvas_id'], outline=original_color, width=2)
                except Exception as e: print(f"Erro ao deselecionar item {slot.get('canvas_id')}: {e}")
            self.selected_slot_id = None
        self.update_button_states()

    def add_slot(self, xa, ya, wa, ha):
        if self.img_cv is None or self.scale == 0:
            messagebox.showerror("Erro", "Imagem não carregada.", parent=self); return
        xr = round(xa / self.scale); yr = round(ya / self.scale)
        wr = round(wa / self.scale); hr = round(ha / self.scale)
        if wr < 5 or hr < 5: print(f"Slot pequeno demais na imagem original ({wr}x{hr})."); return

        img_h_orig, img_w_orig = self.img_cv.shape[:2]
        roi_y_start = max(0, yr); roi_y_end = min(img_h_orig, yr + hr)
        roi_x_start = max(0, xr); roi_x_end = min(img_w_orig, xr + wr)
        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
             messagebox.showwarning("Aviso", "Região inválida calculada.", parent=self); return
        roi_original = self.img_cv[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        if roi_original is None or roi_original.size == 0:
             messagebox.showwarning("Aviso", "Não foi possível extrair ROI.", parent=self); return

        tipo = simpledialog.askstring("Tipo", "'c' para clip ou 'f' para fita:", parent=self)
        if not tipo: return
        tipo = tipo.strip().lower(); slot_type_clean = None
        if tipo.startswith("c"): slot_type_clean = "clip"
        elif tipo.startswith("f"): slot_type_clean = "fita"
        else: messagebox.showerror("Erro", "Tipo inválido.", parent=self); return

        res = pick_color(self, roi_original)
        if res is None: return
        bgr_color, (ht, st, vt) = res

        self.last_slot_id += 1; slot_id = self.last_slot_id; tpl_path = None
        if slot_type_clean == "clip":
            base_name = Path(self.img_path).stem if self.img_path else "modelo"
            tpl_name = f"{base_name}_slot{slot_id}.png"
            tpl_path_obj = TEMPLATE_DIR / tpl_name
            try:
                if roi_original.shape[0] > 0 and roi_original.shape[1] > 0:
                     cv2.imwrite(str(tpl_path_obj), roi_original)
                     tpl_path = tpl_path_obj.name # Armazena só o nome
                     print(f"Template salvo: {tpl_path_obj}")
                else: messagebox.showwarning("Aviso", "ROI inválida para template.", parent=self); tpl_path = None
            except Exception as e: messagebox.showerror("Erro", f"Salvar Template falhou:\n{e}", parent=self); tpl_path = None

        new_slot = {"id": slot_id, "rect": [xr, yr, wr, hr], "type": slot_type_clean,
                    "color": bgr_color, "h": ht, "s": st, "v": vt, "template": tpl_path,
                    "canvas_id": None}
        self.slots.append(new_slot)
        print(f"Slot {slot_id} adicionado.")
        self.redraw_slots()

    def update_slot_data(self, updated_slot_data):
        slot_id_to_update = updated_slot_data.get('id')
        if slot_id_to_update is None: return
        found = False
        for i, slot in enumerate(self.slots):
            if slot['id'] == slot_id_to_update:
                updated_slot_data['canvas_id'] = slot.get('canvas_id')
                self.slots[i] = updated_slot_data; found = True
                print(f"Slot {slot_id_to_update} atualizado.")
                break
        if not found: print(f"Erro: Slot {slot_id_to_update} não encontrado para update."); return
        self.deselect_all_slots()
        self.redraw_slots()

    def clear_slots(self):
        if not self.slots: return
        if messagebox.askyesno("Confirmar", "Remover TODOS os slots?", parent=self):
            self.slots = []; self.last_slot_id = 0
            self.canvas.delete("slot"); self.selected_slot_id = None
            self.update_button_states(); print("Slots removidos.")

    def edit_selected_slot(self):
        if self.selected_slot_id is None: messagebox.showinfo("Aviso", "Nenhum slot selecionado.", parent=self); return
        slot_to_edit = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
        if not slot_to_edit: messagebox.showerror("Erro", f"Dados do slot {self.selected_slot_id} não encontrados.", parent=self); return
        EditSlotDialog(self, slot_to_edit, self)

    def delete_selected_slot(self):
        if self.selected_slot_id is None: messagebox.showinfo("Aviso", "Nenhum slot selecionado.", parent=self); return
        slot_to_delete = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
        if not slot_to_delete: messagebox.showerror("Erro", f"Dados do slot {self.selected_slot_id} não encontrados.", parent=self); self.selected_slot_id = None; self.update_button_states(); return

        if messagebox.askyesno("Confirmar", f"Excluir slot {self.selected_slot_id}?", parent=self):
            if slot_to_delete.get('canvas_id'):
                try: self.canvas.delete(slot_to_delete['canvas_id'])
                except Exception as e: print(f"Aviso: Erro ao remover item do canvas {slot_to_delete['canvas_id']}: {e}")
            self.slots = [s for s in self.slots if s['id'] != self.selected_slot_id]
            if slot_to_delete['type'] == 'clip' and slot_to_delete.get('template'):
                try:
                     tpl_path_to_delete = TEMPLATE_DIR / slot_to_delete['template']
                     if tpl_path_to_delete.exists():
                         if messagebox.askyesno("Excluir Template?", f"Excluir arquivo de template?\n({slot_to_delete['template']})", parent=self):
                              tpl_path_to_delete.unlink(); print(f"Template {tpl_path_to_delete} excluído.")
                except Exception as e: print(f"Erro ao excluir template {slot_to_delete.get('template')}: {e}")
            print(f"Slot {self.selected_slot_id} excluído.")
            self.selected_slot_id = None; self.update_button_states()

    def update_button_states(self):
        has_image = self.img_cv is not None; has_slots = bool(self.slots); slot_is_selected = self.selected_slot_id is not None
        self.save_button.config(state=NORMAL if has_image else DISABLED)
        self.clear_button.config(state=NORMAL if has_slots else DISABLED)
        self.edit_button.config(state=NORMAL if slot_is_selected else DISABLED)
        self.delete_button.config(state=NORMAL if slot_is_selected else DISABLED)

    def save_model(self):
        if self.img_path is None or self.img_cv is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.", parent=self); return
        try: image_path_relative = Path(self.img_path).relative_to(MODEL_DIR.parent); image_path_to_save = str(image_path_relative)
        except ValueError: image_path_to_save = self.img_path
        model_data = {"description": "Modelo Inspeção Visual", "creation_date": datetime.now().isoformat(),
                      "image_path": image_path_to_save, "slots": [] }
        for slot in self.slots:
            slot_copy = slot.copy(); slot_copy.pop('canvas_id', None); model_data["slots"].append(slot_copy)

        save_path = None; overwrite = False
        if self.current_model_path:
            response = messagebox.askyesnocancel("Salvar", f"Sobrescrever modelo?\n({Path(self.current_model_path).name})\n\n(Sim=Sobrescrever, Não=Salvar Como...)", parent=self)
            if response is True: save_path = self.current_model_path; overwrite = True
            elif response is False: save_path = None
            else: return
        if save_path is None:
            default_name = Path(self.img_path).stem + ".json"
            filepath = filedialog.asksaveasfilename(title="Salvar Modelo Como...", initialdir=MODEL_DIR, initialfile=default_name, defaultextension=".json", filetypes=[("Modelo JSON", "*.json")])
            if not filepath: return
            save_path = filepath; overwrite = True
        try:
            with open(save_path, 'w') as f: json.dump(model_data, f, indent=4)
            print(f"Modelo salvo: {save_path}")
            messagebox.showinfo("Sucesso", f"Modelo salvo em:\n{save_path}", parent=self)
            self.current_model_path = save_path
        except Exception as e: messagebox.showerror("Erro", f"Salvar falhou:\n{e}", parent=self)

    def show_help(self):
        help_text = """Editor de Malha:\n\n1. Abrir Imagem: Carrega nova imagem base.\n2. Carregar Modelo: Carrega .json salvo (imagem+slots).\n3. Desenhar Slot: Clique e arraste na imagem.\n4. Selecionar Slot: Clique dentro de um slot.\n5. Editar/Excluir: Use os botões com um slot selecionado.\n6. Salvar Modelo: Salva estado atual em .json."""
        messagebox.showinfo("Ajuda - Editor", help_text, parent=self)


# ---------- Aba Inspeção ------------------------------------------------------
class InspecaoFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=5)
        self.model_data = None       # Dados carregados do JSON {image_path, slots}
        self.img_ref_cv = None       # Imagem de referência (OpenCV BGR) carregada do modelo
        self.img_test_cv = None      # Imagem de teste (OpenCV BGR) carregada pelo usuário
        self.tk_test_img = None      # Imagem de teste para exibição no canvas (Tkinter PhotoImage)
        self.test_img_scale = 1.0    # Escala da imagem de teste no canvas
        self.test_img_path = None    # Caminho da imagem de teste
        self.model_path = None       # Caminho do modelo carregado

        # --- Frame Superior (Controles) ---
        control_frame = ttk.Frame(self)
        control_frame.pack(side=TOP, fill=X, pady=5)

        # Carregar Modelo
        model_frame = ttk.LabelFrame(control_frame, text=" Modelo de Referência ")
        model_frame.pack(side=LEFT, padx=5, fill=Y)
        ttk.Button(model_frame, text="Carregar Modelo (.json)", command=self.browse_load_model, bootstyle="info").pack(pady=2, padx=5, fill=X)
        self.model_label = ttk.Label(model_frame, text="Nenhum modelo carregado.", anchor=NW, justify=LEFT, wraplength=200)
        self.model_label.pack(pady=2, padx=5, fill=X)

        # Carregar Imagem Teste
        test_img_frame = ttk.LabelFrame(control_frame, text=" Imagem de Teste ")
        test_img_frame.pack(side=LEFT, padx=5, fill=Y)
        ttk.Button(test_img_frame, text="Selecionar Imagem Teste", command=self.browse_load_test_image, bootstyle="secondary").pack(pady=2, padx=5, fill=X)
        self.test_img_label = ttk.Label(test_img_frame, text="Nenhuma imagem carregada.", anchor=NW, justify=LEFT, wraplength=200)
        self.test_img_label.pack(pady=2, padx=5, fill=X)

        # Botão Iniciar Inspeção
        action_frame = ttk.LabelFrame(control_frame, text=" Ação ")
        action_frame.pack(side=LEFT, padx=5, fill=Y)
        self.inspect_button = ttk.Button(action_frame, text="Iniciar Inspeção", command=self.run_inspection, state=DISABLED, bootstyle="success")
        self.inspect_button.pack(pady=2, padx=5, fill=BOTH, expand=YES)

        # --- Frame Inferior (Canvas e Logs) ---
        results_frame = ttk.Frame(self)
        results_frame.pack(side=TOP, fill=BOTH, expand=YES, pady=5)

        # Canvas para Imagem de Teste com Resultados
        self.canvas_frame = ttk.LabelFrame(results_frame, text=" Resultado Visual ")
        self.canvas_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0,5))

        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=VERTICAL)
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=HORIZONTAL)
        self.v_scroll.pack(side=RIGHT, fill=Y)
        self.h_scroll.pack(side=BOTTOM, fill=X)

        try: style = ttk.Style.get_instance(); canvas_bg = style.lookup('TFrame', 'background')
        except Exception: canvas_bg = "#333333"
        self.canvas = Canvas(self.canvas_frame, bg=canvas_bg,
                             yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set,
                             highlightthickness=0)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # Área de Logs
        log_frame = ttk.LabelFrame(results_frame, text=" Logs da Inspeção ")
        log_frame.pack(side=RIGHT, fill=Y, padx=(5,0)) # fill Y para ocupar altura
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=10, width=45, state=DISABLED) # Largura ajustada
        # Configurar cores do ScrolledText para tema escuro
        try:
             fg_color = style.lookup('TLabel', 'foreground')
             bg_color = style.lookup('TFrame', 'background') # Usa fundo do frame
             # Pode precisar ajustar cores de seleção também
             self.log_text.config(fg=fg_color, bg=bg_color, insertbackground=fg_color) # Cor do cursor
        except Exception:
             pass # Usa defaults se estilo falhar
        self.log_text.pack(fill=BOTH, expand=YES, pady=5, padx=5)


    def _log(self, message):
        """Adiciona uma mensagem à área de log."""
        self.log_text.config(state=NORMAL)
        self.log_text.insert(END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(END) # Auto-scroll
        self.log_text.config(state=DISABLED)
        print(f"LOG: {message}") # Também printa no console

    def _clear_canvas_overlays(self):
        """Remove apenas os desenhos de resultado do canvas, mantendo a imagem."""
        self.canvas.delete("result_overlay")

    def _clear_all_inspection(self):
        """Limpa estado da aba de inspeção."""
        self.model_data = None
        self.img_ref_cv = None
        self.img_test_cv = None
        self.tk_test_img = None
        self.test_img_scale = 1.0
        self.test_img_path = None
        self.model_path = None
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0,0,0,0))
        self.model_label.config(text="Nenhum modelo carregado.")
        self.test_img_label.config(text="Nenhuma imagem carregada.")
        self.log_text.config(state=NORMAL)
        self.log_text.delete(1.0, END)
        self.log_text.config(state=DISABLED)
        self.inspect_button.config(state=DISABLED)

    def browse_load_model(self):
        """Carrega o arquivo de modelo JSON."""
        filepath = filedialog.askopenfilename(
            title="Carregar Modelo (.json)", initialdir=MODEL_DIR,
            filetypes=[("Modelo JSON", "*.json"), ("Todos", "*.*")]
        )
        if not filepath: return

        try:
            model_path_obj = Path(filepath)
            with open(model_path_obj, 'r') as f:
                loaded_data = json.load(f)

            # Validação básica
            if "image_path" not in loaded_data or "slots" not in loaded_data:
                 messagebox.showerror("Erro", "Arquivo de modelo inválido.", parent=self); return

            # Encontra e carrega imagem de referência
            ref_image_path_str = loaded_data["image_path"]
            ref_image_path = Path(ref_image_path_str)
            if not ref_image_path.is_absolute():
                 candidate_path = model_path_obj.parent / ref_image_path
                 if candidate_path.exists(): ref_image_path = candidate_path
                 elif not ref_image_path.exists():
                     messagebox.showerror("Erro", f"Imagem de referência não encontrada:\n{ref_image_path_str}", parent=self); return

            img_ref = cv2.imread(str(ref_image_path.resolve()))
            if img_ref is None or img_ref.size == 0:
                 messagebox.showerror("Erro", f"Falha ao carregar imagem de referência:\n{ref_image_path}", parent=self); return

            # Armazena dados se tudo OK
            self.model_data = loaded_data
            self.img_ref_cv = img_ref
            self.model_path = str(model_path_obj.resolve())
            self.model_label.config(text=f"Modelo: {model_path_obj.name}")
            self._log(f"Modelo '{model_path_obj.name}' carregado.")
            self.update_inspect_button_state()

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Modelo", f"Ocorreu um erro:\n{e}", parent=self)
            self._clear_all_inspection()


    def browse_load_test_image(self):
        """Carrega a imagem de teste a ser inspecionada."""
        filepath = filedialog.askopenfilename(
            title="Selecionar Imagem de Teste",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")]
        )
        if not filepath: return

        try:
            test_img_path_obj = Path(filepath)
            img_test = cv2.imread(str(test_img_path_obj))
            if img_test is None or img_test.size == 0:
                 messagebox.showerror("Erro", f"Não foi possível carregar a imagem de teste:\n{filepath}", parent=self); return

            self.img_test_cv = img_test
            self.test_img_path = str(test_img_path_obj.resolve())
            self.test_img_label.config(text=f"Teste: {test_img_path_obj.name}")
            self._log(f"Imagem de teste '{test_img_path_obj.name}' carregada.")

            # Exibe a imagem de teste no canvas
            self.canvas.delete("all") # Limpa canvas antes de nova imagem
            self.tk_test_img, self.test_img_scale = cv2_to_tk(self.img_test_cv, PREVIEW_W, PREVIEW_H)
            if self.tk_test_img:
                canvas_img_w = self.tk_test_img.width()
                canvas_img_h = self.tk_test_img.height()
                self.canvas.create_image(0, 0, anchor=NW, image=self.tk_test_img, tags="test_image")
                self.canvas.config(scrollregion=(0, 0, canvas_img_w, canvas_img_h))
            else:
                messagebox.showerror("Erro", "Falha ao converter imagem de teste para exibição.", parent=self)
                self.img_test_cv = None; self.test_img_path = None

            self.update_inspect_button_state()

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Imagem Teste", f"Ocorreu um erro:\n{e}", parent=self)
            # Limpa apenas dados da imagem de teste
            self.img_test_cv = None; self.tk_test_img = None; self.test_img_path = None
            self.test_img_label.config(text="Nenhuma imagem carregada.")
            self.canvas.delete("all"); self.canvas.config(scrollregion=(0,0,0,0))
            self.update_inspect_button_state()


    def update_inspect_button_state(self):
        """Habilita o botão de inspeção se modelo e imagem de teste estiverem carregados."""
        if self.model_data and self.img_ref_cv is not None and self.img_test_cv is not None:
            self.inspect_button.config(state=NORMAL)
        else:
            self.inspect_button.config(state=DISABLED)


    def run_inspection(self):
        """Executa o processo de inspeção completo."""
        if not self.model_data or self.img_ref_cv is None or self.img_test_cv is None:
            messagebox.showerror("Erro", "Carregue o modelo de referência E a imagem de teste antes de inspecionar.", parent=self)
            return

        self._log("--- Iniciando Inspeção ---")
        self._clear_canvas_overlays() # Limpa resultados anteriores do canvas

        # 1. Alinhamento de Imagem
        self._log("Realizando alinhamento ORB...")
        M, _, align_error = find_image_transform(self.img_ref_cv, self.img_test_cv)

        if M is None:
            self._log(f"FALHA no Alinhamento: {align_error}")
            messagebox.showerror("Falha no Alinhamento", f"Não foi possível alinhar as imagens.\nErro: {align_error}", parent=self)
            # Desenha os slots da referência (sem transformação) em cor de erro
            for slot in self.model_data['slots']:
                 xr, yr, wr, hr = slot['rect']
                 # Escala para o canvas da imagem de teste
                 xa, ya = xr * self.test_img_scale, yr * self.test_img_scale
                 wa, ha = wr * self.test_img_scale, hr * self.test_img_scale
                 self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=COLOR_ALIGN_FAIL, width=2, tags="result_overlay")
                 self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"Slot {slot['id']}\nALIGN FAIL", fill=COLOR_ALIGN_FAIL, tags="result_overlay", justify="center")
            return
        else:
             self._log("Alinhamento OK.")

        # 2. Verificação dos Slots
        overall_ok = True
        slots_results = [] # Armazena resultados para resumo

        for slot in self.model_data['slots']:
            self._log(f"Verificando Slot {slot['id']} ({slot['type']})...")
            is_ok, correlation, pixels, corners, bbox, log_msgs = check_slot(self.img_test_cv, slot, M)

            for msg in log_msgs: self._log(f"  -> {msg}") # Log detalhado

            slots_results.append({"id": slot['id'], "ok": is_ok, "corners": corners, "bbox": bbox})
            if not is_ok:
                overall_ok = False

            # 3. Desenha resultado no Canvas
            if corners is not None:
                # Converte cantos (coordenadas da imagem original) para coords do canvas
                canvas_corners = [(int(pt[0] * self.test_img_scale), int(pt[1] * self.test_img_scale)) for pt in corners]
                fill_color = COLOR_PASS if is_ok else COLOR_FAIL
                # Desenha o polígono transformado
                self.canvas.create_polygon(canvas_corners, outline=fill_color, fill="", width=2, tags="result_overlay")
                # Adiciona ID do slot perto do primeiro canto
                self.canvas.create_text(canvas_corners[0][0], canvas_corners[0][1] - 10, # Acima do canto sup-esq
                                          text=f"S{slot['id']}", fill=fill_color, anchor=NW, tags="result_overlay",
                                          font=("Arial", 8, "bold")) # Fonte menor
            elif bbox != [0,0,0,0]: # Se falhou mas tem bbox (ex: fora dos limites)
                 xa, ya = bbox[0] * self.test_img_scale, bbox[1] * self.test_img_scale
                 wa, ha = bbox[2] * self.test_img_scale, bbox[3] * self.test_img_scale
                 self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=COLOR_FAIL, width=1, dash=(4, 2), tags="result_overlay")
                 self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"Slot {slot['id']}\nERROR", fill=COLOR_FAIL, tags="result_overlay", justify="center")

        # 4. Resultado Final
        final_status = "APROVADO" if overall_ok else "REPROVADO"
        self._log(f"--- Inspeção Concluída: {final_status} ---")
        messagebox.showinfo("Resultado da Inspeção", f"A inspeção foi concluída.\nResultado Geral: {final_status}", parent=self)


# ---------- Aplicação Principal -----------------------------------------------
if __name__ == "__main__":
    try: import ttkbootstrap
    except ImportError: print("Erro: ttkbootstrap não encontrado. Instale com: pip install ttkbootstrap"); sys.exit(1)

    # Define o tema escuro aqui
    dark_theme = "cyborg" # Outras opções: darkly, solar, superhero

    try:
        # root = ttk.Window(themename=dark_theme) # Cria janela com tema
        style = ttk.Style(theme=dark_theme)
        root = style.master # Obtem a janela Tk associada ao estilo
        root.title("Editor/Inspetor de Malha v2.1")

        # Tenta definir um tamanho e centralizar
        window_width = 1100 # Aumentado um pouco
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        # Garante que não fique fora da tela
        if center_x < 0: center_x = 0
        if center_y < 0: center_y = 0
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    except Exception as e:
        print(f"Erro ao inicializar com ttkbootstrap theme '{dark_theme}': {e}. Usando Tkinter padrão.")
        root = ttk.Tk()
        root.title("Editor/Inspetor de Malha v2.1 (Tk Default)")
        root.geometry("1100x800")

    # Notebook para as abas
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, padx=10, expand=True, fill=BOTH)

    # Aba Editor
    malha_editor_frame = MalhaFrame(notebook)
    malha_editor_frame.pack(fill=BOTH, expand=YES)
    notebook.add(malha_editor_frame, text=' Editor de Malha ')

    # Aba Inspeção
    inspecao_frame = InspecaoFrame(notebook)
    inspecao_frame.pack(fill=BOTH, expand=YES)
    notebook.add(inspecao_frame, text=' Inspeção ')

    # Garante que janelas OpenCV fechem ao sair
    root.protocol("WM_DELETE_WINDOW", lambda: (cv2.destroyAllWindows(), root.destroy()))

    root.mainloop()