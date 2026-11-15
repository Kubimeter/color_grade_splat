"""
Color grading for Gaussian splats in LichtFeld Studio

Adjusts saturation, hue, brightness etc. for selected Gaussians or whole
splats
"""

import math

import lichtfeld as lf

SH_C0 = 0.28209479177387814
LUMA = (0.2126, 0.7152, 0.0722)

DEFAULTS = {
    "saturation": 1.0,
    "hue": 0.0,
    "exposure": 0.0,
    "brightness": 0.0,
    "contrast": 1.0,
    "temperature": 0.0,
    "tint": 0.0,
    "colorize_amount": 0.0,
    "colorize_hex": "#BFD9F2",
}

# key, label, slider range, lowest value allowed when typed in
SLIDERS = [
    ("saturation", "Saturation", 0.0, 2.0, 0.0),
    ("hue", "Hue (°)", -180.0, 180.0, None),
    ("exposure", "Exposure (EV)", -2.0, 2.0, None),
    ("brightness", "Brightness", -0.3, 0.3, None),
    ("contrast", "Contrast", 0.5, 1.5, 0.0),
    ("temperature", "Temperature (blue↔yellow)", -1.0, 1.0, None),
    ("tint", "Tint (green↔magenta)", -1.0, 1.0, None),
]


# ----------------------------------------------------------------------------
# Color
# ----------------------------------------------------------------------------

def parse_hex(text):
    digits = text.strip().lstrip("#")
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    if len(digits) != 6:
        return None
    try:
        return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None


def hue_saturation_matrix(saturation, hue_degrees):
    # Same matrices as the CSS filters hue-rotate() and saturate()
    # Rotation around the gray axis, brightness stays roughly the same
    a = math.radians(hue_degrees)
    c, s = math.cos(a), math.sin(a)
    hue = [
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ]
    k = saturation
    sat = [
        [0.213 + 0.787 * k, 0.715 - 0.715 * k, 0.072 - 0.072 * k],
        [0.213 - 0.213 * k, 0.715 + 0.285 * k, 0.072 - 0.072 * k],
        [0.213 - 0.213 * k, 0.715 - 0.715 * k, 0.072 + 0.928 * k],
    ]
    return [[sum(sat[i][m] * hue[m][j] for m in range(3)) for j in range(3)] for i in range(3)]


def grade(rgb, p):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    gain = 2.0 ** p["exposure"]
    r = r * (gain * (1.0 + 0.2 * p["temperature"]))
    g = g * (gain * (1.0 - 0.2 * p["tint"]))
    b = b * (gain * (1.0 - 0.2 * p["temperature"]))

    m = hue_saturation_matrix(p["saturation"], p["hue"])
    r, g, b = (
        r * m[0][0] + g * m[0][1] + b * m[0][2],
        r * m[1][0] + g * m[1][1] + b * m[1][2],
        r * m[2][0] + g * m[2][1] + b * m[2][2],
    )

    contrast, offset = p["contrast"], p["brightness"]
    r = (r - 0.5) * contrast + 0.5 + offset
    g = (g - 0.5) * contrast + 0.5 + offset
    b = (b - 0.5) * contrast + 0.5 + offset

    target = parse_hex(p["colorize_hex"])
    amount = p["colorize_amount"]
    if target and amount > 0.0:
        # Target color is scaled to each Gaussian's brightness, so the structure is kept and only the hue shifts
        target_luma = max(sum(w * c for w, c in zip(LUMA, target)), 1e-4)
        luma = r * LUMA[0] + g * LUMA[1] + b * LUMA[2]
        r = r + (luma * (target[0] / target_luma) - r) * amount
        g = g + (luma * (target[1] / target_luma) - g) * amount
        b = b + (luma * (target[2] / target_luma) - b) * amount

    return lf.Tensor.stack([r, g, b], 1)


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------

def node_name(node):
    name = getattr(node, "name", None)
    return name if name is not None else node.get("name")


def splat_nodes_with_offsets(scene):
    # The selection mask is one long array covering all splat nodes back to back, in scene order
    nodes, offset = [], 0
    for node in scene.get_nodes():
        if not str(node.type).upper().endswith("SPLAT"):
            continue
        count = int(node.gaussian_count)
        if count:
            nodes.append((node, offset, count))
            offset += count
    return nodes, offset


def find_targets(scene, use_selection):
    """Returns (name, mask); mask is None when the whole node is meant"""
    nodes, total = splat_nodes_with_offsets(scene)

    if not use_selection:
        selected = set(lf.get_selected_node_names())
        targets = [(node_name(n), None) for n, _, _ in nodes if node_name(n) in selected]
        return targets, None if targets else "No splat node selected in the scene tree"

    mask = scene.selection_mask
    if mask is None or not scene.has_selection():
        return [], "No Gaussians selected"
    if mask.numel != total:
        return [], "Selection doesn't match the splat nodes. Please reload the scene"

    targets = []
    for node, offset, count in nodes:
        part = mask[offset:offset + count] != 0
        if int(part.sum().item()):
            targets.append((node_name(node), part))
    return targets, None if targets else "No Gaussians selected"


def splat_data(scene, name):
    node = scene.get_node(name)
    return node.splat_data() if node else None


def write_sh0(scene, name, values):
    data = splat_data(scene, name)
    if data is None or data.num_points != values.shape[0]:
        return False
    target = data.sh0_raw
    target[:] = values
    return True


# ----------------------------------------------------------------------------
# Editing
# ----------------------------------------------------------------------------

class GradeSession:
    """Keeps the original colors so the sliders always start from the initial
    state and changes don't stack up"""

    def __init__(self):
        self.entries = []
        self.count = 0
        self.dirty = False

    @classmethod
    def start(cls, use_selection):
        scene = lf.get_scene()
        if scene is None:
            return None, "No scene loaded"
        targets, error = find_targets(scene, use_selection)
        if error:
            return None, error

        session = cls()
        for name, mask in targets:
            node = scene.get_node(name)
            if node is None or node.read_only:
                lf.log.warn(f"[ColorGrade] '{name}' is read-only and will be skipped")
                continue
            data = node.splat_data()
            if data is None:
                continue

            n = data.num_points
            original = data.sh0_raw.clone()
            entry = {
                "name": name,
                "n": n,
                "original": original,
                "rgb": original.reshape([n, 3]) * SH_C0 + 0.5,
                "mask": None,
            }
            if mask is not None:
                entry["mask"] = mask.reshape([n, 1]).expand([n, 3]).contiguous()
                session.count += int(mask.sum().item())
            else:
                session.count += n
            session.entries.append(entry)

        if not session.entries:
            return None, "Nothing to edit"
        return session, None

    def preview(self, params):
        scene = lf.get_scene()
        for e in self.entries:
            rgb = grade(e["rgb"], params)
            if e["mask"] is not None:
                rgb = lf.Tensor.where(e["mask"], rgb, e["rgb"])
            sh0 = ((rgb - 0.5) / SH_C0).reshape([e["n"], 1, 3]).contiguous()
            if not write_sh0(scene, e["name"], sh0):
                return f"'{e['name']}' was changed in the meantime"
        self.dirty = True
        scene.notify_changed()
        return None

    def discard(self):
        scene = lf.get_scene()
        if scene is None or not self.dirty:
            return
        for e in self.entries:
            write_sh0(scene, e["name"], e["original"])
        scene.notify_changed()
        self.dirty = False

    def commit(self):
        scene = lf.get_scene()
        if scene is None or not self.dirty:
            return "No changes to commit"

        changes = []
        for e in self.entries:
            data = splat_data(scene, e["name"])
            if data is not None:
                changes.append((e["name"], e["original"], data.sh0_raw.clone()))

        def restore(use_original):
            current = lf.get_scene()
            if current is None:
                return
            for name, before, after in changes:
                write_sh0(current, name, before if use_original else after)
            current.notify_changed()

        size = sum(before.numel * 8 for _, before, _ in changes)
        lf.undo.push("Color grading", lambda: restore(True), lambda: restore(False),
                     source="splat_color_grade", estimated_bytes=size)
        self.dirty = False
        return f"Color grading committed for {self.count:,} Gaussians"


# Lives at module level so on_unload can revert an open preview
_session = None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

class ColorGradePanel(lf.ui.Panel):
    id = "splat_color_grade.panel"
    label = "Color Grading"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 150

    def __init__(self):
        self.params = dict(DEFAULTS)
        self.use_selection = True
        self.live = True
        self.status = ""

    @classmethod
    def poll(cls, context):
        return lf.has_scene()

    def draw(self, ui):
        active = _session is not None
        ui.heading("Color Grading")

        ui.begin_disabled(active)
        _, self.use_selection = ui.checkbox("Selected Gaussians only", self.use_selection)
        _, self.live = ui.checkbox("Live preview", self.live)
        ui.end_disabled()

        if active:
            ui.text_disabled("Selection is locked until you commit or discard")
        elif not self.use_selection:
            ui.text_disabled("Applies to the splat nodes selected in the scene tree")

        ui.separator()
        changed = False
        for key, label, low, high, minimum in SLIDERS:
            changed |= self.slider(ui, key, label, low, high, minimum)

        if ui.collapsing_header("Colorize", default_open=False):
            changed |= self.hex_input(ui)
            changed |= self.slider(ui, "colorize_amount", "Strength", 0.0, 1.0, 0.0)

        if changed and self.live:
            self.run(self.update_preview)

        ui.separator()
        if ui.button_styled("Commit" if self.live or active else "Apply", "primary"):
            self.run(self.apply)
        ui.same_line()
        if ui.button_styled("Discard" if active else "Reset sliders", "secondary"):
            self.reset()

        if self.status:
            ui.text_wrapped(self.status)
        ui.text_disabled("Undo after committing: Ctrl+Z")

    def slider(self, ui, key, label, low, high, minimum):
        old = self.params[key]

        _, value = ui.slider_float(f"{label}##{key}_slider", old, low, high)
        ui.same_line()
        ui.push_item_width(70)
        typed, typed_value = ui.input_float(f"##{key}_input", old, 0.0, 0.0, "%.3f")
        ui.pop_item_width()

        if typed:
            value = typed_value if minimum is None else max(minimum, typed_value)
        self.params[key] = value
        return value != old

    def hex_input(self, ui):
        old_color = parse_hex(self.params["colorize_hex"])
        _, text = ui.input_text("Target color (hex)##colorize_hex", self.params["colorize_hex"])
        self.params["colorize_hex"] = text

        color = parse_hex(text)
        if color is None:
            ui.text_disabled("Format: #RRGGBB, e.g. #A8C8E0")
            return False
        ui.text_colored("■■■■■■  Preview", (*color, 1.0))
        return color != old_color

    def run(self, action):
        try:
            action()
        except Exception as error:
            self.status = f"Error: {error}"
            lf.log.error(f"[ColorGrade] {error}")

    def update_preview(self):
        global _session
        if _session is None:
            _session, error = GradeSession.start(self.use_selection)
            if error:
                self.status = error
                return

        error = _session.preview(self.params)
        if error:
            _session.discard()
            _session = None
            self.status = error
        else:
            self.status = f"Previewing {_session.count:,} Gaussians"

    def apply(self):
        global _session
        if _session is None:
            if self.live:
                self.status = "No changes to commit"
                return
            self.update_preview()
            if _session is None:
                return

        self.status = _session.commit()
        _session = None
        # After committing, the new state becomes the baseline
        if self.live:
            self.params = dict(DEFAULTS)

    def reset(self):
        global _session
        if _session is not None:
            _session.discard()
            _session = None
            self.status = "Discarded"
        else:
            self.status = ""
        self.params = dict(DEFAULTS)


def on_load():
    lf.register_class(ColorGradePanel)


def on_unload():
    global _session
    if _session is not None:
        try:
            _session.discard()
        except Exception:
            pass
        _session = None
    lf.unregister_class(ColorGradePanel)