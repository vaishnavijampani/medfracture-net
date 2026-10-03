import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

from config import config

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    NAVY = RGBColor(26, 54, 93)      # #1A365D
    STEEL = RGBColor(43, 108, 176)   # #2B6CB0
    DARK = RGBColor(45, 55, 72)       # #2D3748
    WHITE = RGBColor(255, 255, 255)
    LIGHT_BG = RGBColor(247, 250, 252) # #F7FAFC

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="IEEE RESEARCH PROTOTYPE DEMONSTRATION"):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(1.1))
        shape.fill.solid()
        shape.fill.fore_color.rgb = NAVY
        shape.line.fill.background()

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.5)
        tf.margin_top = Inches(0.15)

        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = STEEL

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = WHITE

    # SLIDE 1: Title Slide
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    txBox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.33), Inches(4.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "MAJOR B.TECH FINAL-YEAR PROJECT PRESENTATION"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = STEEL

    p1 = tf.add_paragraph()
    p1.text = "MedFracture-Net: A Multi-Task Deep Learning Framework for Real-Time Bone Fracture Detection, Localization, and Severity Assessment"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "IEEE Transactions on Medical Imaging Target Architecture"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_before = Pt(15)

    p3 = tf.add_paragraph()
    p3.text = "Author: Student Research Team  |  Guide: Project Advisor  |  Department of Computer Science & Engineering"
    p3.font.size = Pt(13)
    p3.font.color.rgb = STEEL
    p3.space_before = Pt(30)

    # SLIDE 2: Problem Statement & Clinical Motivation
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Problem Statement & Clinical Motivation")
    
    tb2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    bullets2 = [
        ("Clinical Diagnostic Bottleneck", "Manual interpretation of X-ray radiographs is subjective, labor-intensive, and prone to human diagnostic error during high-volume emergency department shifts."),
        ("Failure of Classical Edge Detectors", "Existing literature relies heavily on classical Canny edge detection, Otsu thresholding, or Adaptive Gaussian filters, which fail under low X-ray contrast, non-uniform illumination, and soft-tissue overlaps."),
        ("Micro-Fracture Diagnostic Gaps", "Hairline and non-displaced fractures are frequently missed in early-stage X-rays, leading to delayed orthopedic stabilization and improper healing."),
        ("Lack of Clinical Interpretability", "Existing black-box deep learning models lack explainable visual features, making radiologists reluctant to rely on AI outputs without visual proof.")
    ]
    for title, body in bullets2:
        p = tf2.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)
        
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 3: Project Objectives
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Project Objectives & Scope")

    tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    objs = [
        ("1. Real-Time Detection", "Binary verification of fracture presence vs. intact healthy bone cortex."),
        ("2. Sub-Pixel Localization", "Instance segmentation of exact crack boundaries using U-Net++ nested skip connections."),
        ("3. Morphological Classification", "Categorize fractures into 6 types: Transverse, Oblique, Spiral, Comminuted, Hairline, Greenstick."),
        ("4. Quantitative Severity Estimation", "Calculate physical fracture length in millimeters (mm) using spatial pixel calibration."),
        ("5. Explainable AI (XAI)", "Generate Grad-CAM++ feature activation maps to visually audit network decision rationale."),
        ("6. Clinical PDF & Web Deployment", "Deliver a real-time Streamlit web workstation and dynamic PDF report generator.")
    ]
    for title, body in objs:
        p = tf3.add_paragraph()
        p.text = f"{title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = STEEL
        p.space_before = Pt(10)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 4: Literature Review & Gap Analysis
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Literature Review & Base Paper Gap Analysis")

    rows, cols = 5, 4
    table_shape = slide4.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    table = table_shape.table

    headers = ["Feature / Dimension", "Base IEEE Reference Paper", "Existing DL Approaches", "Proposed MedFracture-Net"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE

    table_data = [
        ["Core Algorithm", "Classical Canny & Otsu Edge Filters", "Standard Single U-Net", "U-Net++ & ResNet-34 Multi-Task"],
        ["Target Application", "Rigid Industrial Workpieces", "Simple 2D Bone Masks", "Complex Biological Radiographs"],
        ["Classification Accuracy", "94.12%", "92.0% - 95.5%", "98.70% (+4.58% Improvement)"],
        ["Clinical Explainability", "None (Threshold Heuristics)", "None (Black Box CNN)", "Grad-CAM++ Saliency Heatmaps"]
    ]
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx+1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = NAVY if c_idx == 3 else DARK
                if c_idx == 3:
                    p.font.bold = True

    # SLIDE 5: System Architecture & Workflow
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "MedFracture-Net System Architecture")

    tb5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    arch_steps = [
        ("Phase 1: Radiograph Preprocessing", "Contrast Limited Adaptive Histogram Equalization (CLAHE) with tile grid equalization and bilateral noise removal."),
        ("Phase 2: Sub-Pixel Segmentation", "Nested U-Net++ architecture capturing fine structural trabecular crack lines."),
        ("Phase 3: Multi-Class Classification", "Deep ResNet-34 feature extraction network for 6 morphological categories."),
        ("Phase 4: Explainable AI (XAI)", "Grad-CAM++ gradient attribution engine highlighting feature focus areas."),
        ("Phase 5: Severity & Report Generation", "cv2.minAreaRect geometric millimeter calculator and dynamic PDF report generator.")
    ]
    for title, body in arch_steps:
        p = tf5.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(14)
        run.font.color.rgb = DARK

    # SLIDE 6: Preprocessing & CLAHE Enhancement
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Radiograph Preprocessing & CLAHE Enhancement")

    tb6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    clahe_points = [
        ("Contrast Limited Adaptive Histogram Equalization (CLAHE)", "Divides X-ray into 8x8 contextual regions with clip limit = 3.0 to equalize local intensity distribution without over-amplifying background noise."),
        ("Bilateral Noise Filtering", "Preserves sharp bone cortical boundaries while eliminating high-frequency sensor noise."),
        ("Albumentations Augmentation Pipeline", "Applies random rotations (15°), horizontal flips, shift-scale-rotate, and Gaussian noise to guarantee model robustness across multi-center X-ray machines.")
    ]
    for title, body in clahe_points:
        p = tf6.add_paragraph()
        p.text = f"• {title}:\n  "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = STEEL
        p.space_before = Pt(14)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 7: Deep Learning Methodology (U-Net++ & ResNet-34)
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Deep Learning Methodology: U-Net++ & ResNet-34")

    tb7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    dl_points = [
        ("Nested U-Net++ Segmentation Architecture", "Incorporates re-designed dense skip pathways (conv0_0 to conv0_4) to bridge semantic gaps between encoder and decoder feature maps, preventing edge detail loss."),
        ("ResNet-34 Classification Backbone", "Leverages residual shortcut connections (x + F(x)) to eliminate vanishing gradients during deep feature extraction across 6 morphological classes."),
        ("Joint Multi-Task Loss Function", "Minimizes combined objective: L_Total = 0.5 * L_Dice + 0.3 * L_FocalCE + 0.2 * L_BCE")
    ]
    for title, body in dl_points:
        p = tf7.add_paragraph()
        p.text = f"• {title}:\n  "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 8: Explainable AI & Physical Severity Estimation
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Explainable AI (Grad-CAM++) & Physical Severity Calibration")

    tb8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    xai_points = [
        ("Grad-CAM++ Feature Attribution", "Calculates higher-order partial derivatives of classification scores with respect to final ResNet feature maps, producing pixel-accurate heatmaps that audit AI diagnostic focus."),
        ("Spatial Millimeter Calibration", "Physical Length (mm) = Pixel Extent x Spatial Calibration Factor (0.15 mm/px)."),
        ("Clinical Severity Grading", "Mild (< 5.0 mm), Moderate (5.0 - 15.0 mm), Severe (> 15.0 mm).")
    ]
    for title, body in xai_points:
        p = tf8.add_paragraph()
        p.text = f"• {title}:\n  "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = STEEL
        p.space_before = Pt(14)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 9: Experimental Results & Comparative Performance
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Empirical Metric Benchmark & Baseline Comparison (10k Cohort)")

    rows, cols = 6, 6
    t_shape9 = slide9.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    t9 = t_shape9.table

    headers9 = ["Method / Architecture", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "Dice Score"]
    for i, h in enumerate(headers9):
        cell = t9.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE

    table_data9 = [
        ["Scratch Ablation (Random Init)", "57.88%", "57.80%", "100.0%", "73.20%", "0.125"],
        ["Otsu Thresholding", "91.25%", "90.12%", "89.75%", "89.93%", "0.880"],
        ["Adaptive Gaussian Filter", "93.48%", "92.85%", "92.10%", "92.47%", "0.910"],
        ["Canny Edge Detector (Base)", "94.12%", "93.56%", "93.05%", "93.30%", "0.920"],
        ["MedFracture-Net (Patient-Independent)", "99.49%", "99.75%", "99.37%", "99.56%", "0.984"]
    ]
    for r_idx, row in enumerate(table_data9):
        for c_idx, val in enumerate(row):
            cell = t9.cell(r_idx+1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12.5)
                p.font.color.rgb = NAVY if c_idx == 0 or c_idx == 1 else DARK
                if r_idx == 4:
                    p.font.bold = True

    # SLIDE 10: Training Convergence Telemetry & Patient Independence Audit
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Empirical Validation & Zero Patient Data Leakage Audit")

    tb10 = slide10.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    telemetry_points = [
        ("Audited Patient Leakage Resolution", "Identified 96.25% patient ID leakage in naive Kaggle folder splits caused by rotations (1-rotated1.jpg). Resolved via Patient-Independent Group Splitting across 121 unique patients with 0% patient leakage."),
        ("Strictly Held-Out Generalization", "Evaluated on 1,377 radiographs across 25 unseen patient entities. Correctly predicted 1,370 out of 1,377 radiographs (99.49% empirical accuracy, 95% Wilson CI: [98.95%, 99.75%])."),
        ("Diagnostic Sensitivity & Specificity", "Achieved 99.37% clinical sensitivity (792/797 fractures detected) and 99.66% specificity (578/580 normal bones identified), yielding 0.9998 ROC-AUC."),
        ("Backbone Transfer Learning Ablation", "Untrained scratch model achieved only 57.88% accuracy (0.5056 AUC), confirming that ImageNet-pretrained representations are vital for robust radiographic boundary discernment.")
    ]
    for title, body in telemetry_points:
        p = tf10.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(14)
        run.font.color.rgb = DARK

    # SLIDE 11: Real-Time Web Application & PDF Generation
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Real-Time Workstation & Automated PDF Reporting")

    tb11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    app_points = [
        ("Streamlit Interactive Dashboard", "Real-time web interface supporting drag-and-drop DICOM/PNG X-ray uploads."),
        ("Anatomical Auto-Detection", "Morphological profile classifier automatically identifies Forearm vs. Leg structure."),
        ("Ultra-Fast Latency", "Inference latency of 31.4 ms (> 30 FPS execution rate)."),
        ("Automated PDF Diagnostic Export", "Generates formal medical diagnostic reports formatted with finding summary, spatial metrics, and visual overlays.")
    ]
    for title, body in app_points:
        p = tf11.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = STEEL
        p.space_before = Pt(14)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # SLIDE 12: Conclusion & Future Work
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Conclusion & Future Work")

    tb12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    conc_points = [
        ("Summary of Achievements", "Engineered MedFracture-Net, achieving 98.70% accuracy, 0.97 Dice score, and 31.4 ms real-time inference latency."),
        ("Novel Core Contributions", "Integrated U-Net++, ResNet-34, Grad-CAM++ XAI, CLAHE enhancement, and physical millimeter severity estimation into a unified framework."),
        ("Future Research Extension 1", "3D Volumetric CT Expansion: Extending 2D X-ray pipeline to 3D DICOM CT volumes via 3D Swin-UNETR."),
        ("Future Research Extension 2", "Edge Mobile Hardware Deployment: Quantizing models via TensorRT for edge execution on ambulance digital X-ray units.")
    ]
    for title, body in conc_points:
        p = tf12.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = DARK

    # Save presentation
    output_path = os.path.join(config.BASE_DIR, "MedFracture_Net_Presentation.pptx")
    prs.save(output_path)
    print(f"SUCCESS: Presentation saved to {output_path}")

if __name__ == "__main__":
    create_deck()
