"""DeepInspect AI — Multi-Type Bottle Inspection (Mobile + Camera + PatchCore).

نسخة موسعة تدعم:
- PatchCore (EfficientNet-B0) للتعامل مع أي نوع زجاجة
- ConvAutoencoder الأصلي للمقارنة الأكاديمية
- كاميرا الهاتف مباشرة
- إضافة أنواع زجاجات جديدة بـ 5 صور فقط
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as T

from src.models.autoencoder import ConvAutoencoder, load_compatible_state_dict
from src.models.patchcore import PatchCoreDetector


# ==============================================================================
# 1. Page Config
# ==============================================================================
st.set_page_config(
    page_title="DeepInspect AI 🔍",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .verdict-pass {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #ecfdf5; border-radius: 16px; padding: 1.5rem;
        text-align: center; font-size: 1.6rem; font-weight: 900;
        margin: 1rem 0; box-shadow: 0 8px 25px rgba(16,185,129,0.4);
    }
    .verdict-reject {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fef2f2; border-radius: 16px; padding: 1.5rem;
        text-align: center; font-size: 1.6rem; font-weight: 900;
        margin: 1rem 0; box-shadow: 0 8px 25px rgba(239,68,68,0.5);
    }
    .score-box {
        background: rgba(0,0,0,0.05); border-radius: 12px;
        padding: 0.8rem 1rem; font-family: monospace; font-size: 1rem;
        margin: 0.3rem 0; border: 1px solid rgba(0,0,0,0.1);
    }
    .type-badge {
        display:inline-block; background: #1d4ed8; color: white;
        padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button { width: 100%; padding: 0.6rem 1rem;
        font-size: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Load Models
# ==============================================================================
PATCHCORE_BANK = "models/patchcore_bank.pkl"
CAE_CHECKPOINT = "models/best_model.pth"


@st.cache_resource(show_spinner="⏳ تحميل نماذج الذكاء الاصطناعي...")
def load_patchcore() -> Optional[PatchCoreDetector]:
    """Load PatchCore detector if bank exists."""
    detector = PatchCoreDetector(device="cpu", subsample_ratio=0.5)
    if Path(PATCHCORE_BANK).exists():
        detector.load(PATCHCORE_BANK)
        return detector
    return None


@st.cache_resource(show_spinner="⏳ تحميل نموذج ConvAutoencoder...")
def load_cae() -> Optional[ConvAutoencoder]:
    if not Path(CAE_CHECKPOINT).exists():
        return None
    model = ConvAutoencoder(in_channels=3, input_size=(256, 256),
                            channels=[32, 64, 128, 256], latent_dim=128)
    ckpt = torch.load(CAE_CHECKPOINT, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        load_compatible_state_dict(model, ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        load_compatible_state_dict(model, ckpt)
    model.eval()
    return model


patchcore = load_patchcore()
cae_model = load_cae()


# ==============================================================================
# 3. Header
# ==============================================================================
st.markdown("# 🔍 DeepInspect AI")
st.markdown("**فحص عيوب الزجاجات بالذكاء الاصطناعي** — يدعم جميع أنواع القوارير 🍾")

# Status bar
col_s1, col_s2 = st.columns(2)
with col_s1:
    if patchcore and patchcore.enrolled_types:
        st.success(f"✅ PatchCore: {len(patchcore.enrolled_types)} نوع مسجّل")
    else:
        st.warning("⚠️ PatchCore: لا يوجد أنواع مسجّلة بعد")
with col_s2:
    if cae_model:
        st.success("✅ ConvAutoencoder: محمّل (MVTec AD)")
    else:
        st.info("ℹ️ ConvAutoencoder: غير متاح")
st.markdown("---")


# ==============================================================================
# 4. Sidebar Settings
# ==============================================================================
with st.sidebar:
    st.header("⚙️ إعدادات الفحص")

    # Model selector
    available_engines = []
    if patchcore and patchcore.enrolled_types:
        available_engines.append("🧠 PatchCore (متعدد الأنواع — مُوصى به)")
    if cae_model:
        available_engines.append("📐 ConvAutoencoder (MVTec AD فقط)")

    if not available_engines:
        available_engines = ["لا يوجد نموذج مُحمَّل"]

    engine_choice = st.selectbox("محرك الفحص:", available_engines)
    use_patchcore = "PatchCore" in engine_choice

    if use_patchcore and patchcore:
        bottle_type = st.selectbox(
            "نوع الزجاجة:",
            options=patchcore.enrolled_types,
        )
    else:
        bottle_type = ""

    heatmap_alpha = st.slider("شفافية الخريطة الحرارية", 0.1, 0.9, 0.55, 0.05)
    show_details = st.checkbox("تفاصيل الفحص", value=True)

    st.markdown("---")
    st.info(
        "📱 **للهاتف:** تأكد أن هاتفك على نفس شبكة WiFi ثم افتح:\n\n"
        "`http://192.168.88.157:8501`"
    )


# ==============================================================================
# 5. Inference Functions
# ==============================================================================
def run_patchcore(image: Image.Image, btype: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run PatchCore inference. Returns (score, threshold, heatmap_rgb, orig_np)."""
    orig_np = np.array(image.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    score, heatmap_rgb = patchcore.predict(image, btype)
    threshold = patchcore.get_threshold(btype, default=1.5)
    overlay = cv2.addWeighted(orig_np, 1.0 - heatmap_alpha, heatmap_rgb, heatmap_alpha, 0)
    return score, threshold, heatmap_rgb, overlay


def run_cae(image: Image.Image) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Run ConvAutoencoder inference. Returns (score, threshold, heatmap, overlay, recon_np)."""
    resized = image.convert("RGB").resize((256, 256), Image.BILINEAR)
    orig_np = np.array(resized, dtype=np.uint8)
    t = T.ToTensor()(resized).unsqueeze(0)
    with torch.no_grad():
        recon = cae_model(t)
        diff = (t - recon) ** 2
        residual = diff.mean(dim=1).squeeze().numpy()
        recon_np = (recon.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    smoothed = cv2.GaussianBlur(residual, (7, 7), 4.0, 4.0)
    score = float(np.mean(smoothed))
    threshold = 0.0040
    res_min, res_max = smoothed.min(), smoothed.max()
    if res_max > res_min:
        norm = ((smoothed - res_min) / (res_max - res_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(smoothed, dtype=np.uint8)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_np, 1 - heatmap_alpha, heatmap_rgb, heatmap_alpha, 0)
    return score, threshold, heatmap_rgb, overlay, recon_np


def show_verdict(score: float, threshold: float, engine: str) -> bool:
    """Display big verdict banner. Returns True if anomaly."""
    is_anomaly = score >= threshold
    if is_anomaly:
        st.markdown(
            '<div class="verdict-reject">🚨 معيب — DEFECT DETECTED<br>'
            '<span style="font-size:0.9rem;opacity:0.85">يُنصح بإبعاد هذه القارورة</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict-pass">✅ سليم — NORMAL PASS<br>'
            '<span style="font-size:0.9rem;opacity:0.85">القارورة مطابقة للمعيار</span></div>',
            unsafe_allow_html=True,
        )
    score_color = "#ef4444" if is_anomaly else "#10b981"
    st.markdown(
        f'<div class="score-box">درجة الشذوذ: '
        f'<strong style="color:{score_color}">{score:.4f}</strong>'
        f' | العتبة: <strong>{threshold:.4f}</strong>'
        f' | المحرك: <span class="type-badge">{engine}</span></div>',
        unsafe_allow_html=True,
    )
    return is_anomaly


# ==============================================================================
# 6. Main Tabs
# ==============================================================================
tab_inspect, tab_enroll = st.tabs([
    "🔍 فحص قارورة",
    "➕ تسجيل نوع جديد",
])


# ---- TAB 1: Inspection -------------------------------------------------------
with tab_inspect:
    st.markdown("### 📸 مصدر الصورة")
    sub_cam, sub_upload, sub_demo = st.tabs([
        "📷 كاميرا", "📁 رفع صورة", "🧪 عينات تجريبية",
    ])

    input_pil: Optional[Image.Image] = None
    input_source = ""

    with sub_cam:
        st.markdown(
            "📱 صوّر **فوهة القارورة أو جسمها** مباشرة.\n\n"
            "> للحصول على أفضل النتائج: إضاءة جيدة وخلفية فاتحة وزاوية علوية."
        )
        cam = st.camera_input("📷 التقط صورة القارورة")
        if cam:
            input_pil = Image.open(cam)
            input_source = "📷 كاميرا مباشرة"

    with sub_upload:
        up = st.file_uploader("ارفع صورة القارورة (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if up:
            input_pil = Image.open(up)
            input_source = f"📁 {up.name}"

    with sub_demo:
        demo_opts = {
            "🟢 زجاجة سليمة (MVTec)": "data/demo_samples/real_good_bottle.png",
            "🟢 عينة تدريب سليمة": "data/demo_samples/train_good_000.png",
            "🔴 زجاجة مكسورة (MVTec)": "data/demo_samples/real_broken_bottle.png",
            "🔴 زجاجة ملوثة (MVTec)": "data/demo_samples/real_contaminated_bottle.png",
        }

        # Add any coca-cola / green bottle samples if present
        extra = {
            "🍾 زجاجة كوكاكولا (خارج التوزيع)": "data/demo_samples/full_coca_cola_bottle.png",
            "🍾 زجاجة خضراء (خارج التوزيع)": "data/demo_samples/full_green_bottle.jpg",
        }
        for label, path in extra.items():
            if Path(path).exists():
                demo_opts[label] = path

        sel = st.selectbox("اختر عينة:", list(demo_opts.keys()))
        if st.button("🔍 افحص هذه العينة", key="btn_demo"):
            p = demo_opts[sel]
            if Path(p).exists():
                input_pil = Image.open(p)
                input_source = f"🧪 {Path(p).name}"
            else:
                st.error(f"الملف غير موجود: {p}")

    # ---- Run Inference -------------------------------------------------------
    if input_pil is not None:
        st.markdown("---")
        st.caption(f"المصدر: {input_source}")
        st.image(input_pil, caption="الصورة المُدخلة", use_container_width=True)

        with st.spinner("⚡ جاري الفحص..."):
            if use_patchcore and patchcore and bottle_type:
                score, threshold, heatmap_rgb, overlay = run_patchcore(input_pil, bottle_type)
                engine_label = f"PatchCore · {bottle_type}"
                is_anomaly = show_verdict(score, threshold, engine_label)

                if show_details:
                    st.markdown("---")
                    st.markdown("#### 🔬 تفاصيل تحليل الصورة (PatchCore)")
                    orig_np = np.array(
                        input_pil.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8
                    )
                    c1, c2 = st.columns(2)
                    c1.image(orig_np, caption="الصورة الأصلية", use_container_width=True)
                    c2.image(heatmap_rgb, caption="خريطة المسافات (PatchCore)", use_container_width=True)
                    st.image(overlay, caption="التحديد البصري للعيب", use_container_width=True)

            elif not use_patchcore and cae_model:
                score, threshold, heatmap_rgb, overlay, recon_np = run_cae(input_pil)
                engine_label = "ConvAutoencoder (MVTec)"
                is_anomaly = show_verdict(score, threshold, engine_label)

                if show_details:
                    st.markdown("---")
                    st.markdown("#### 🔬 تفاصيل (ConvAutoencoder)")
                    orig_np = np.array(
                        input_pil.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.image(orig_np, caption="الأصلية", use_container_width=True)
                    c2.image(recon_np, caption="إعادة البناء", use_container_width=True)
                    c3.image(overlay, caption="الخريطة الحرارية", use_container_width=True)
            else:
                st.warning("يرجى تحديد نموذج فحص من القائمة الجانبية.")

        # Camera tips
        if input_source.startswith("📷"):
            st.info(
                "📌 **نصائح لتحسين دقة التصوير:**\n"
                "- صوّر من **الأعلى مباشرة** على خلفية فاتحة.\n"
                "- إضاءة جيدة بدون ظل أو انعكاس.\n"
                "- المسافة المثالية: 10-25 سم.\n"
                "- إذا كانت النتيجة غير دقيقة، أضف نوع قاروتك عبر تبويب **تسجيل نوع جديد**."
            )
    else:
        st.markdown("---")
        st.info("👆 اختر مصدر الصورة من التبويبات أعلاه للبدء.")

        # Preview demo samples
        st.markdown("#### 🧪 أمثلة على الكشف:")
        pc1, pc2, pc3, pc4 = st.columns(4)
        previews = [
            ("data/demo_samples/real_good_bottle.png", "🟢 سليمة"),
            ("data/demo_samples/train_good_000.png", "🟢 سليمة"),
            ("data/demo_samples/real_broken_bottle.png", "🔴 مكسورة"),
            ("data/demo_samples/real_contaminated_bottle.png", "🔴 ملوثة"),
        ]
        for col, (path, label) in zip([pc1, pc2, pc3, pc4], previews):
            if Path(path).exists():
                col.image(Image.open(path), caption=label, use_container_width=True)


# ---- TAB 2: Enroll new bottle type -------------------------------------------
with tab_enroll:
    st.markdown("### ➕ تسجيل نوع زجاجة جديد")
    st.markdown(
        "يمكنك تعليم النموذج نوع زجاجة جديد تماماً بدون إعادة التدريب!\n\n"
        "**كيف يعمل؟** ارفع **5 صور أو أكثر** من زجاجات سليمة من نفس النوع، "
        "وسيقوم النظام ببناء مكتبة مرجعية لها."
    )

    new_type_name = st.text_input(
        "اسم نوع الزجاجة الجديد (بالعربية أو الإنجليزية):",
        placeholder="مثال: زجاجة_بيرة، plastic_water_bottle، علبة_عصير",
    )

    st.markdown("**ارفع 5 صور أو أكثر من الزجاجات السليمة (بدون عيوب):**")
    enrolled_files = st.file_uploader(
        "صور الزجاجات السليمة (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="enroll_uploader",
    )

    # Optional: also enroll via camera
    st.markdown("**أو التقط صوراً مباشرة بالكاميرا:**")
    cam_enroll = st.camera_input("📷 التقط صورة زجاجة سليمة للتسجيل", key="cam_enroll")

    enroll_images: List[Image.Image] = []
    if enrolled_files:
        enroll_images += [Image.open(f).convert("RGB") for f in enrolled_files]
    if cam_enroll:
        enroll_images.append(Image.open(cam_enroll).convert("RGB"))

    if enroll_images:
        st.info(f"تم تحميل {len(enroll_images)} صورة جاهزة للتسجيل.")
        # Show thumbnails
        thumb_cols = st.columns(min(len(enroll_images), 5))
        for col, img in zip(thumb_cols, enroll_images[:5]):
            col.image(img, use_container_width=True)
        if len(enroll_images) > 5:
            st.caption(f"... و{len(enroll_images) - 5} صور إضافية")

    if st.button(
        f"✅ تسجيل '{new_type_name}' الآن",
        disabled=(not new_type_name.strip() or len(enroll_images) < 1),
        use_container_width=True,
    ):
        if len(enroll_images) < 3:
            st.warning("يُنصح برفع 3 صور على الأقل للحصول على نتائج دقيقة.")

        with st.spinner(f"⏳ جاري بناء المكتبة المرجعية لـ '{new_type_name}'..."):
            # Reload or create detector
            if patchcore is None:
                # Create new detector instance and store in session
                st.session_state["patchcore_instance"] = PatchCoreDetector(
                    device="cpu", subsample_ratio=0.5
                )

            # Use session detector or main detector
            det = st.session_state.get("patchcore_instance", patchcore)
            if det is None:
                det = PatchCoreDetector(device="cpu", subsample_ratio=0.5)
                if Path(PATCHCORE_BANK).exists():
                    det.load(PATCHCORE_BANK)

            det.enroll(new_type_name.strip(), enroll_images, verbose=False)
            det.calibrate_threshold(new_type_name.strip(), enroll_images)
            Path(PATCHCORE_BANK).parent.mkdir(parents=True, exist_ok=True)
            det.save(PATCHCORE_BANK)

        st.success(
            f"✅ تم تسجيل **'{new_type_name}'** بنجاح!\n\n"
            f"الأنواع المتاحة الآن: {', '.join(det.enrolled_types)}\n\n"
            f"أعد تشغيل التطبيق (Ctrl+R) ثم اختر هذا النوع من القائمة الجانبية."
        )
        st.balloons()

    # Show currently enrolled types
    st.markdown("---")
    st.markdown("#### 📋 الأنواع المسجّلة حالياً:")
    if patchcore and patchcore.enrolled_types:
        for t in patchcore.enrolled_types:
            thresh = patchcore.get_threshold(t)
            st.markdown(f"- **{t}** — عتبة القرار: `{thresh:.4f}`")
    else:
        st.info("لم يتم تسجيل أي نوع زجاجة بعد. استخدم النموذج ConvAutoencoder بدلاً من ذلك.")

    st.markdown("---")
    st.markdown(
        "#### 💡 نصائح للحصول على أفضل النتائج:\n"
        "- **5-15 صورة** من زوايا وإضاءات مختلفة قليلاً تعطي أفضل دقة.\n"
        "- صوّر نفس النوع من القارورة من **نفس الزاوية** التي ستستخدمها لاحقاً.\n"
        "- **لا تُدرج صوراً فيها عيوب** في مجموعة التسجيل.\n"
        "- يمكنك تسجيل أنواع لا نهاية لها بدون إعادة تدريب!"
    )
