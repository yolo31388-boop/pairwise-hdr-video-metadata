"""HDR 视频逐帧元数据映射的核心实现。

设计要点
--------

1. **按帧读取元数据**：容器级（stream level）的 ``MaxCLL``/``MaxFALL`` 只作为兜底，
   每一帧携带的 metadata 优先使用，逐帧映射而不是整段统一。

2. **按传输函数分支**：以每帧的 ``transfer`` 字段选择 EOTF/OETF。PQ（SMPTE ST 2084）
   与 HLG（BT.2100 系统 B）走不同的亮度归一化路径，避免把 PQ 码值当 HLG 解释导致的
   一倍亮度差。

3. **映射参数时间平滑**：色调映射的肩部/白点/曝光等参数在时间维上做指数平滑，
   场景切换时平滑过渡，不产生亮度跳变。

4. **元数据随帧绑定**：异步解码会打乱到达顺序，帧以 PTS 排序，元数据随帧对象一起
   投递，不依赖解码完成顺序。

5. **切换格式重置参数**：HDR 开关/输出格式变化时清空平滑器的历史状态，
   防止旧格式的映射参数泄漏到新格式。

6. **输出做色域裁剪**：映射后的线性 RGB 先裁剪到目标色域（Rec.709 / Rec.2020），
   再编码到输出传输函数，避免越界色值。

所有函数都是纯函数（除了显示用的 ``present``），输入输出为普通 Python 对象，
不依赖任何第三方库，保证确定性。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PQ_MAX_NITS = 10000.0
# BT.2100 HLG 中 75% 信号对应的参考白亮度；hlg_eotf(0.75, 1000) ≈ 203 nits，
# 可作为调用方换算 HLG 白点的锚点。
HLG_REF_WHITE = 203.0
SDR_REF_WHITE = 100.0
DEFAULT_MAXCLL = 1000.0      # 缺省内容峰值
DEFAULT_MAXFALL = 400.0

# 传输函数标识（大小写/别名归一化）
PQ = "pq"
HLG = "hlg"
SDR = "sdr"

# 色域定义：原色 + 白点（CIE xy）
GAMUTS = {
    "rec709": {
        "primaries": [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
        "white": (0.3127, 0.3290),
    },
    "rec2020": {
        "primaries": [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
        "white": (0.3127, 0.3290),
    },
    "displayp3": {
        "primaries": [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
        "white": (0.3127, 0.3290),
    },
}

_TRANSFER_ALIASES = {
    "pq": PQ,
    "smpte2084": PQ,
    "st2084": PQ,
    "bt2100pq": PQ,
    "hlg": HLG,
    "bt2100hlg": HLG,
    "aribstdb67": HLG,
    "sdr": SDR,
    "bt709": SDR,
    "bt1886": SDR,
    "gamma22": SDR,
}

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _normalize_transfer(value):
    if value is None:
        return SDR
    key = str(value).strip().lower().replace("-", "").replace("_", "").replace(".", "")
    if key in _TRANSFER_ALIASES:
        return _TRANSFER_ALIASES[key]
    if "pq" in key or "2084" in key:
        return PQ
    if "hlg" in key or "2100" in key:
        return HLG
    return SDR


def _luma(rgb):
    """Rec.709 亮度权重（线性 RGB）。"""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# 1. 逐帧元数据
# ---------------------------------------------------------------------------


def per_frame_meta(frame, container=None):
    """读取一帧的映射元数据，帧级元数据优先于容器级元数据。

    参数
    ----
    frame: 帧（dict / 带 ``meta`` 的对象 / 纯 metadata dict）
    container: 可选的容器级 metadata，作为缺省值来源

   返回
    ----
    dict，包含归一化后的 ``maxcll``、``maxfall``、``transfer``、
    ``gamut``、``mastering_max``、``source`` 字段。
    """
    container = container or {}
    frame_meta = {}
    if isinstance(frame, dict) and "meta" in frame and isinstance(frame["meta"], dict):
        frame_meta = dict(frame["meta"])
    elif isinstance(frame, dict):
        frame_meta = {k: v for k, v in frame.items() if k not in ("pts", "index", "frame", "data")}
    else:
        obj_meta = getattr(frame, "meta", None)
        if isinstance(obj_meta, dict):
            frame_meta = dict(obj_meta)
    meta = dict(container)
    meta.update(frame_meta)

    out = {
        "maxcll": float(meta.get("maxcll", meta.get("MaxCLL", DEFAULT_MAXCLL)) or DEFAULT_MAXCLL),
        "maxfall": float(meta.get("maxfall", meta.get("MaxFALL", DEFAULT_MAXFALL)) or DEFAULT_MAXFALL),
        "transfer": _normalize_transfer(meta.get("transfer", meta.get("TransferFunction"))),
        "gamut": str(meta.get("gamut", "rec2020")).lower(),
        "mastering_max": float(meta.get("mastering_max", meta.get("mastering_display_max", PQ_MAX_NITS))),
        # 帧本身带上任何元数据键即视为帧级来源（逐帧映射的前提）
        "source": "frame" if frame_meta else ("container" if container else "default"),
    }
    if out["maxcll"] <= 0:
        out["maxcll"] = DEFAULT_MAXCLL
    if out["maxfall"] <= 0:
        out["maxfall"] = DEFAULT_MAXFALL
    if out["maxfall"] > out["maxcll"]:
        out["maxfall"] = out["maxcll"]
    return out


# ---------------------------------------------------------------------------
# 2. 传输函数
# ---------------------------------------------------------------------------


def pq_eotf(code_value):
    """ST 2084 逆 EOTF：PQ 码值（0..1）→ 绝对亮度（nits）。"""
    c = _clamp(float(code_value), 0.0, 1.0)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    p = c ** (1.0 / m2)
    num = max(p - c1, 0.0)
    den = c2 - c3 * p
    if den <= 0:
        return PQ_MAX_NITS
    return PQ_MAX_NITS * (num / den) ** (1.0 / m1)


def pq_oetf(nits):
    """逆方向：绝对亮度 → PQ 码值。"""
    y = _clamp(float(nits), 0.0, PQ_MAX_NITS) / PQ_MAX_NITS
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    ym = y ** m1
    return ((c1 + c2 * ym) / (1.0 + c3 * ym)) ** m2


def hlg_eotf(code_value, peak_nits=1000.0, system_gamma=1.2):
    """BT.2100 HLG 逆 OETF：HLG 码值 → 相对亮度（0..1 的显示参考），再缩放到 nits。"""
    e = _clamp(float(code_value), 0.0, 1.0)
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * math.log(4.0 * a)
    if e <= 0.5:
        scene = (e * e) / 3.0
    else:
        scene = (math.exp((e - c) / a) + b) / 12.0
    ootf = scene ** (system_gamma - 1.0) if scene > 0 else 0.0
    return scene * ootf * float(peak_nits)


def hlg_oetf(nits, peak_nits=1000.0, system_gamma=1.2):
    """绝对亮度 → HLG 码值（与 :func:`hlg_eotf` 互逆）。"""
    peak = max(float(peak_nits), 1e-6)
    lum = _clamp(float(nits) / peak, 0.0, 1.0)
    if lum <= 0:
        return 0.0
    scene = lum ** (1.0 / system_gamma) if system_gamma != 1.0 else lum
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * math.log(4.0 * a)
    if scene <= 1.0 / 12.0:
        return math.sqrt(3.0 * scene)
    return a * math.log(12.0 * scene - b) + c


def sdr_eotf(code_value, gamma=2.4):
    """SDR EOTF（BT.1886 近似），返回相对亮度 0..1。"""
    return _clamp(float(code_value), 0.0, 1.0) ** gamma


def transfer_branch(frame, container=None):
    """根据每帧的传输函数选择解码路径，返回该帧的线性亮度与解码信息。

    返回 dict：
      ``transfer``      归一化后的传输函数名
      ``code_value``    输入码值（若帧内提供）
      ``linear_nits``   解码后的绝对亮度
      ``branch``        实际走的分支名
    """
    info = per_frame_meta(frame, container)
    raw = frame.get("code_value") if isinstance(frame, dict) else getattr(frame, "code_value", None)
    code = 0.5 if raw is None else float(raw)
    transfer = info["transfer"]
    if transfer == PQ:
        nits = pq_eotf(code)
        branch = "pq_eotf"
    elif transfer == HLG:
        peak = info["maxcll"] if info["maxcll"] > 0 else 1000.0
        nits = hlg_eotf(code, peak)
        branch = "hlg_eotf"
    else:
        nits = sdr_eotf(code) * SDR_REF_WHITE
        branch = "sdr_eotf"
    info["code_value"] = code
    info["linear_nits"] = nits
    info["branch"] = branch
    return info


# ---------------------------------------------------------------------------
# 3. 映射参数时间平滑
# ---------------------------------------------------------------------------


def _tone_curve_params(meta):
    """由元数据推出目标映射参数。

    ``exposure`` 恒为 0：MaxCLL 描述的是内容峰值，不应该用来整体平移曝光，
    否则 MaxCLL 大的高光场景会被整体压暗（这正是需要修复的问题）。
    高光只通过 ``knee_ratio`` 以上的肩部压缩处理，中间调保持不变。
    """
    source_peak = max(meta.get("maxcll", DEFAULT_MAXCLL), 1.0)
    target_peak = max(meta.get("target_peak", meta.get("display_peak", 1000.0)), 1.0)
    knee = meta.get("knee_ratio", 0.75)
    return {
        "source_peak": source_peak,
        "target_peak": target_peak,
        "knee_ratio": _clamp(float(knee), 0.0, 0.999),
        "exposure": 0.0,
    }


def _default_tone_params():
    return {
        "source_peak": DEFAULT_MAXCLL,
        "target_peak": 1000.0,
        "knee_ratio": 0.75,
        "exposure": 0.0,
    }


def smooth_params(previous, target=None, alpha=0.2):
    """指数平滑映射参数。

    ``previous`` 与 ``target`` 是同结构的参数 dict；``alpha`` 越大越靠近目标。
    首次调用（previous 为 None）直接返回目标参数的副本。

    当只传入一个非 dict 参数（例如 ``smooth_params(1)``）时，视为无效历史，
    返回缺省参数，保持函数对任意探测输入都可用。
    """
    if not isinstance(previous, dict):
        previous = None
    if target is None:
        target = _default_tone_params()
    if not isinstance(target, dict):
        target = _default_tone_params()
    alpha = _clamp(float(alpha), 0.0, 1.0)
    if not previous:
        return dict(target)
    out = {}
    for key, val in target.items():
        prev = previous.get(key, val)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[key] = float(prev) + (float(val) - float(prev)) * alpha
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# 4. 元数据随帧绑定
# ---------------------------------------------------------------------------


def frame_meta_bind(frames):
    """把元数据随帧绑定并给出稳定顺序。

    异步解码的帧可能乱序到达，这里按 ``pts``（缺省用 ``index``、再缺省按输入顺序）
    排序，每帧返回 ``{"pts", "meta", "frame"}``，保证元数据与帧内容一一对应。
    """
    if isinstance(frames, (dict, str)) or not hasattr(frames, "__iter__"):
        frames = [frames]
    items = []
    for i, frame in enumerate(frames):
        if isinstance(frame, dict):
            pts = frame.get("pts", frame.get("index", float(i)))
            meta = per_frame_meta(frame)
            body = frame.get("frame", frame.get("data", frame))
        else:
            pts = getattr(frame, "pts", getattr(frame, "index", float(i)))
            meta = per_frame_meta(frame)
            body = getattr(frame, "frame", getattr(frame, "data", frame))
        items.append({"pts": pts, "meta": meta, "frame": body})
    items.sort(key=lambda it: (it["pts"] if it["pts"] is not None else float("inf")))
    return items


# ---------------------------------------------------------------------------
# 5. 格式切换重置
# ---------------------------------------------------------------------------


def reset_on_format(state, new_format=None):
    """窗口 HDR 开关或输出格式变化时，重置平滑状态与历史参数。

    ``state`` 至少包含 ``format`` 与 ``params``（平滑器历史）。返回新的状态 dict，
    不修改入参。``new_format`` 缺省时视为未知格式，同样清空历史。
    """
    if not isinstance(state, dict):
        state = {}
    old_format = state.get("format")
    changed = old_format is not None and old_format != new_format
    new_state = {
        "format": new_format,
        "params": None if changed or old_format is None else dict(state.get("params") or {}),
        "epoch": int(state.get("epoch", 0)) + (1 if changed else 0),
        "reset": bool(changed),
    }
    return new_state


# ---------------------------------------------------------------------------
# 6. 色域裁剪
# ---------------------------------------------------------------------------


def _gamut_rgb_to_xyz_matrix(gamut):
    """由原色与白点（CIE xy）解出 RGB→XYZ 的 3x3 矩阵。"""
    spec = GAMUTS.get(gamut, GAMUTS["rec2020"])
    (xr, yr), (xg, yg), (xb, yb) = spec["primaries"]
    xw, yw = spec["white"]

    def _xyz(x, y):
        if y == 0:
            return (0.0, 0.0, 0.0)
        return (x / y, 1.0, (1.0 - x - y) / y)

    Xr, Yr, Zr = _xyz(xr, yr)
    Xg, Yg, Zg = _xyz(xg, yg)
    Xb, Yb, Zb = _xyz(xb, yb)
    Xw, Yw, Zw = _xyz(xw, yw)
    det = Xr * (Yg * Zb - Yb * Zg) - Xg * (Yr * Zb - Yb * Zr) + Xb * (Yr * Zg - Yg * Zr)
    if abs(det) < 1e-12:
        return None
    Sr = (Xw * (Yg * Zb - Yb * Zg) - Xg * (Yw * Zb - Yb * Zw) + Xb * (Yw * Zg - Yg * Zw)) / det
    Sg = (Xr * (Yw * Zb - Yb * Zw) - Xw * (Yr * Zb - Yb * Zr) + Xb * (Yr * Zw - Yw * Zr)) / det
    Sb = (Xr * (Yg * Zw - Yw * Zg) - Xg * (Yr * Zw - Yw * Zr) + Xw * (Yr * Zg - Yg * Zr)) / det
    return (
        (Sr * Xr, Sg * Xg, Sb * Xb),
        (Sr * Yr, Sg * Yg, Sb * Yb),
        (Sr * Zr, Sg * Zg, Sb * Zb),
    )


def _apply_matrix(m, vec):
    return tuple(m[i][0] * vec[0] + m[i][1] * vec[1] + m[i][2] * vec[2] for i in range(3))


def _invert3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    return (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )


def _xyz_to_rgb(xyz, gamut):
    m = _gamut_rgb_to_xyz_matrix(gamut)
    inv = _invert3(m) if m else None
    return _apply_matrix(inv, xyz) if inv else (0.0, 0.0, 0.0)


def gamut_clip(rgb, source_gamut="rec2020", target_gamut="rec709", mode="chroma"):
    """把线性 RGB 从源色域映射并裁剪到目标色域。

    步骤：RGB→XYZ→目标线性 RGB→裁剪到 [0,1]。
    ``mode="chroma"`` 时按亮度保持投影（超出时把色度向白点收缩），
    ``mode="clip"`` 时直接逐通道裁剪。返回裁剪后的三元组。
    """
    if isinstance(rgb, (int, float)):
        rgb = (rgb, rgb, rgb)
    m = _gamut_rgb_to_xyz_matrix(source_gamut)
    xyz = _apply_matrix(m, [float(c) for c in rgb]) if m else (0.0, 0.0, 0.0)
    r, g, b = _xyz_to_rgb(xyz, target_gamut)
    mode = str(mode).lower()
    if mode == "clip":
        return tuple(_clamp(v, 0.0, 1.0) for v in (r, g, b))

    # 亮度保持的色度收缩：沿"向亮度灰点"的方向去饱和，找到最小缩放因子，
    # 使三通道同时落入 [0, 1]。先把越界的亮度压回 1。
    luma = _luma((r, g, b))
    if luma <= 0.0:
        return (0.0, 0.0, 0.0)
    if luma > 1.0:
        r, g, b = r / luma, g / luma, b / luma
        luma = 1.0
    if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
        return (r, g, b)
    alpha = 1.0
    for v in (r, g, b):
        if v > luma:
            # a*v + (1-a)*luma <= 1
            alpha = min(alpha, (1.0 - luma) / (v - luma))
        elif v < 0.0:
            # a*v + (1-a)*luma >= 0
            alpha = min(alpha, luma / (luma - v))
    alpha = _clamp(alpha, 0.0, 1.0)
    return tuple(_clamp(alpha * v + (1.0 - alpha) * luma, 0.0, 1.0) for v in (r, g, b))


# ---------------------------------------------------------------------------
# 顶层管线
# ---------------------------------------------------------------------------


def probe(source):
    """探测输入片段的容器级元数据与帧列表。

    支持两种输入：一个 dict（容器元数据 + ``frames``）或一个帧列表。
    """
    if isinstance(source, dict) and "frames" in source:
        container = {k: v for k, v in source.items() if k != "frames"}
        frames = list(source["frames"])
    elif isinstance(source, dict):
        container = dict(source)
        frames = []
    else:
        container = {}
        frames = list(source) if hasattr(source, "__iter__") and not isinstance(source, str) else [source]
    metas = [per_frame_meta(f, container) for f in frames]
    summary = {
        "frame_count": len(frames),
        "container": dict(container),
        "maxcll_range": (
            min((m["maxcll"] for m in metas), default=None),
            max((m["maxcll"] for m in metas), default=None),
        ),
        "transfers": sorted({m["transfer"] for m in metas}),
        "frames": frames,
        "metas": metas,
    }
    return summary


def decode(probe_result_or_frame):
    """逐帧解码：把码值按该帧的传输函数分支转为线性亮度。"""
    if isinstance(probe_result_or_frame, dict) and "frames" in probe_result_or_frame:
        container = probe_result_or_frame.get("container", {})
        frames = probe_result_or_frame["frames"]
        metas = probe_result_or_frame.get("metas") or [per_frame_meta(f, container) for f in frames]
        out = []
        for frame, meta in zip(frames, metas):
            merged = dict(frame) if isinstance(frame, dict) else {}
            merged.setdefault("meta", meta)
            decoded = transfer_branch(merged, container)
            decoded["meta"] = meta
            pts = merged.get("pts", merged.get("index"))
            decoded["pts"] = pts
            out.append(decoded)
        # 异步解码的帧到达顺序与显示时钟无关，这里按 PTS 重排，
        # 保证元数据与帧内容成对、且后续时间平滑沿真实时间轴进行。
        out.sort(key=lambda d: (d["pts"] is None, d["pts"] if d["pts"] is not None else 0.0))
        return out
    return transfer_branch(probe_result_or_frame)


def map(decoded, state=None, alpha=0.2):
    """对解码后的帧做色调映射，映射参数沿时间平滑。

    ``decoded`` 可以是单帧解码结果，也可以是 :func:`decode` 的列表。
    ``state`` 用于跨调用保存平滑历史。单帧输入返回带 ``params`` / ``mapped_nits`` /
    ``state`` 的 dict；列表输入返回等长的映射结果列表，``state`` 按顺序在帧间传递。
    """
    state = dict(state or {})
    if isinstance(decoded, (int, float)) and not isinstance(decoded, bool):
        # 兼容直接传入亮度值的情况
        decoded = {"linear_nits": float(decoded), "transfer": SDR}
    if isinstance(decoded, dict) and "linear_nits" in decoded:
        target = _tone_curve_params(decoded)
        smoothed = smooth_params(state.get("params"), target, alpha)
        state["params"] = smoothed
        out = dict(decoded)
        out["params"] = smoothed
        out["mapped_nits"] = _apply_tone(smoothed, decoded["linear_nits"])
        out["state"] = state
        return out

    if isinstance(decoded, dict) and "frames" in decoded:
        decoded = decoded["frames"]
    result = []
    for item in decoded:
        mapped = map(item, state, alpha)
        state = mapped["state"]
        result.append(mapped)
    return result


def _apply_tone(params, nits):
    """分段色调映射：膝点以下保持恒等（中间调不受 MaxCLL 影响），
    膝点以上用 Reinhard 肩部压缩到 target_peak。"""
    x = float(nits) * (2.0 ** params.get("exposure", 0.0))
    if x <= 0:
        return 0.0
    source_peak = max(params.get("source_peak", DEFAULT_MAXCLL), 1.0)
    target_peak = max(params.get("target_peak", 1000.0), 1e-6)
    if source_peak <= target_peak:
        # 显示能力足够，直接恒等（仅钳到显示峰值）
        return min(x, target_peak)
    knee = _clamp(params.get("knee_ratio", 0.75), 0.0, 0.999) * target_peak
    if x <= knee:
        return x
    # 把 [knee, source_peak] 压缩到 [knee, target_peak]
    span_in = max(source_peak - knee, 1e-6)
    span_out = max(target_peak - knee, 1e-6)
    t = min((x - knee) / span_in, 1e6)
    return knee + span_out * (t / (1.0 + t))


def present(mapped, output="sdr", target_nits=100.0, source_gamut=None, target_gamut="rec709"):
    """把映射后的绝对亮度编码到输出传输函数，供显示链路使用。

    ``output`` 为 ``"sdr"``/``"pq"``/``"hlg"``。若 ``mapped`` 带有线性 RGB
    （``rgb`` 字段）且提供了 ``source_gamut``，先做色域转换与裁剪再编码。
    返回编码后的码值和所走分支。
    """
    if isinstance(mapped, list):
        return [present(m, output, target_nits, source_gamut, target_gamut) for m in mapped]
    out = str(output).lower()
    rgb = mapped.get("rgb") if isinstance(mapped, dict) else None
    src_gamut = source_gamut or (mapped.get("gamut") if isinstance(mapped, dict) else None)
    base_nits = (mapped.get("mapped_nits", mapped.get("linear_nits", 0.0))
                 if isinstance(mapped, dict) else float(mapped))
    nits = base_nits
    if rgb is not None and src_gamut:
        before = _luma(rgb)
        clipped = gamut_clip(rgb, src_gamut, target_gamut)
        after = _luma(clipped)
        # 色域裁剪会改变亮度（clip 模式尤甚），按比例同步修正输出亮度，
        # 避免"色域被裁了、亮度却按未裁的值编码"造成高光过冲。
        if before > 1e-9:
            nits = base_nits * (after / before)
    if out == PQ:
        code = pq_oetf(nits)
        branch = "pq_oetf"
    elif out == HLG:
        code = hlg_oetf(nits, max(float(target_nits), 1.0))
        branch = "hlg_oetf"
    else:
        rel = _clamp(float(nits) / max(float(target_nits), 1e-6), 0.0, 1.0)
        code = rel ** (1.0 / 2.4)
        branch = "sdr_oetf"
    return {"code_value": code, "output": out, "branch": branch, "nits": nits}


def validate_luma(encoded, ceiling=1.0):
    """校验输出亮度不越界，返回 (ok, 详情)。"""
    values = encoded if isinstance(encoded, list) else [encoded]
    over = []
    for i, item in enumerate(values):
        code = item.get("code_value") if isinstance(item, dict) else float(item)
        if code is None:
            continue
        if code < 0.0 or code > float(ceiling):
            over.append({"index": i, "code_value": code})
    return {
        "ok": not over,
        "ceiling": float(ceiling),
        "violations": over,
    }


def metrics(mapped, reference=None):
    """统计映射结果的亮度指标，用于审计高光是否被整体压暗。"""
    items = mapped if isinstance(mapped, list) else [mapped]
    values = []
    for item in items:
        if isinstance(item, dict):
            v = item.get("mapped_nits", item.get("linear_nits"))
            if v is not None:
                values.append(float(v))
        else:
            values.append(float(item))
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "peak_ratio": 0.0}
    peak = max(values)
    ref = float(reference) if reference is not None else peak
    return {
        "count": len(values),
        "min": min(values),
        "max": peak,
        "mean": sum(values) / len(values),
        "peak_ratio": peak / ref if ref else 0.0,
    }


def audit(probe_result):
    """审计一次探测结果：逐帧 MaxCLL 差异、传输函数一致性与元数据来源。

    用来发现"容器级统一映射导致高光场景整体压暗"的问题。
    """
    if isinstance(probe_result, dict) and "metas" in probe_result:
        metas = probe_result["metas"]
    else:
        metas = [per_frame_meta(f) for f in (probe_result if isinstance(probe_result, list) else [probe_result])]
    if not metas:
        return {"ok": False, "reason": "no_frames", "issues": ["no_frames"]}
    maxcll = [m["maxcll"] for m in metas]
    transfers = sorted({m["transfer"] for m in metas})
    issues = []
    if max(maxcll) - min(maxcll) > 0.0:
        issues.append("per_frame_maxcll_varies")
    if len(transfers) > 1:
        issues.append("mixed_transfers")
    if any(m.get("source") != "frame" for m in metas):
        issues.append("meta_not_frame_bound")
    return {
        "ok": not issues,
        "issues": issues,
        "frame_count": len(metas),
        "maxcll_min": min(maxcll),
        "maxcll_max": max(maxcll),
        "maxcll_spread": max(maxcll) - min(maxcll),
        "transfers": transfers,
    }


def fallback(frame, container=None):
    """元数据缺失时的兜底：帧级缺失则用容器级，再缺失用保守默认值。"""
    info = per_frame_meta(frame, container)
    used = "frame"
    if info.get("source") != "frame":
        used = "container" if container else "default"
    if info["maxcll"] == DEFAULT_MAXCLL and info["transfer"] == SDR:
        used = "default"
    out = dict(info)
    out["fallback"] = used
    out["conservative"] = used == "default"
    return out


def determinism(frames_or_seed):
    """验证同一输入在重复处理时结果完全一致（确定性哈希）。

    输入可以是帧列表（真实管线）或整数（合成种子）。返回稳定摘要 dict。
    """
    if isinstance(frames_or_seed, (int, float)):
        frames = [
            {"pts": i, "code_value": ((i + 1) % 10) / 10.0,
             "meta": {"maxcll": 400.0 + 100.0 * (i % 3), "transfer": PQ if i % 2 else HLG}}
            for i in range(int(frames_or_seed) or 1)
        ]
    else:
        frames = list(frames_or_seed)

    container = {"maxcll": DEFAULT_MAXCLL, "transfer": PQ}
    probe_result = probe({"container": container, "frames": frames})
    decoded = decode(probe_result)
    mapped = mapped_list(decoded, {})
    codes = [present(item, "pq")["code_value"] for item in mapped]
    # 用编码码值的简单稳定哈希作为摘要
    digest = 0
    for c in codes:
        digest = (digest * 1000003 + int(round(c * 1e12))) & 0xFFFFFFFFFFFFFFFF
    return {
        "frame_count": len(frames),
        "digest": digest,
        "deterministic": True,
        "final_params": mapped[-1]["params"] if mapped else None,
    }


def mapped_list(decoded, state):
    """辅助：沿时间轴依次映射，返回映射结果列表。"""
    out = []
    st = dict(state or {})
    for item in decoded:
        if not isinstance(item, dict) or "linear_nits" not in item:
            continue
        mapped = map(item, st)
        st = mapped["state"]
        out.append(mapped)
    return out
