import cv2
import numpy as np
import streamlit as st

st.title("PhytoScan AI 🌿🔬")
st.subheader("Multi-Condition Plant Stress & Disease Analyzer (SDG 8)")

camera_file = st.camera_input("Take a photo of the leaf sample")

if camera_file is not None:
  # 1. Load Image
  bytes_data = camera_file.getvalue()
  cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

  # 2. Convert to HSV Color Space and Grayscale
  hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
  gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

  # 3. Plant Matter Mask (Captures both Healthy Green & Dry Yellow/Brown leaves)
  # Hue 10-25 captures dry/yellow/brown leaves; Hue 25-85 captures healthy green leaves
  lower_plant = np.array([10, 30, 30])
  upper_plant = np.array([85, 255, 255])
  mask = cv2.inRange(hsv_img, lower_plant, upper_plant)
  plant_percentage = (
      cv2.countNonZero(mask) / (cv_img.shape[0] * cv_img.shape[1])
  ) * 100

  # Re-calculate green percentage specifically for metrics display
  lower_green = np.array([25, 40, 40])
  upper_green = np.array([85, 255, 255])
  green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
  green_percentage = (
      cv2.countNonZero(green_mask) / (cv_img.shape[0] * cv_img.shape[1])
  ) * 100

  # --- Plant Validation Check (Blocks faces / non-plant objects) ---
  if plant_percentage < 10.0:
    st.error(
        "❌ **Error: No Plant Detected!** Please scan a valid plant leaf instead"
        " of a face or non-plant object."
    )
  else:
    # 4. Vascular Hydration (Saturation Index)
    avg_saturation = np.mean(hsv_img[:, :, 1])

    # 5. Spot/Lesion Detection (Checking for dark fungal spots or necrosis)
    _, thresh = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY_INV)
    spots_count = (
        cv2.countNonZero(thresh) / (cv_img.shape[0] * cv_img.shape[1])
    ) * 100

    # 6. Classification Logic & Actionable Recommendations
    st.write("---")
    st.write("### 📊 Diagnostic Results:")
    st.metric(
        label="Chlorophyll Density (% Green)", value=f"{green_percentage:.2f}%"
    )
    st.metric(label="Vascular Hydration Index", value=f"{avg_saturation:.1f} / 255")
    st.metric(label="Surface Defect / Spot Index", value=f"{spots_count:.1f}%")

    # Decision tree: Check Healthy -> Drought/Stress -> Fungal -> Early Sub-clinical
    if green_percentage > 55 and spots_count < 15:
      st.success(
          "🟢 **Diagnosis: HEALTHY** — Normal chlorophyll and vascular hydration"
          " detected.\n\n"
          "**Maintenance:** Maintain the current watering schedule and ensure"
          " adequate sunlight exposure aligned with the specific plant"
          " species' requirements."
      )
    elif green_percentage < 35 or avg_saturation < 80:
      st.warning(
          "🟡 **Diagnosis: DROUGHT / NUTRIENT STRESS** — Widespread chlorotic"
          " fading and internal moisture loss detected.\n\n"
          "**Immediate Action:** Move the plant to a shaded, indirect-light"
          " area to reduce immediate transpirational water loss.\n\n"
          "**Watering Protocol:** Apply deep watering via soil saturation"
          " (allow water to drain completely to avoid root rot). Avoid shallow,"
          " daily water splashes that only wet the topsoil.\n\n"
          "**Nutrient Support:** If the chlorosis persists after rehydration,"
          " apply a balanced, water-soluble NPK"
          " (Nitrogen-Phosphorus-Potassium) fertilizer to replenish depleted"
          " macro-nutrients."
      )
    elif spots_count > 25:
      st.error(
          "🟤 **Diagnosis: LEAF SPOT / FUNGAL INFECTION** — High concentration"
          " of localized dark necrotic lesions detected.\n\n"
          "**Immediate Action:** Isolate the affected plant immediately to"
          " prevent spores from spreading to neighboring vegetation.\n\n"
          "**Mechanical Removal:** Carefully prune and dispose of severely"
          " infected leaves using sterilized shears (wipe blades with alcohol"
          " between cuts). Never leave infected leaves in the soil pot.\n\n"
          "**Treatment:** Apply an organic fungicide or a natural copper/neem"
          " oil spray to the remaining foliage, focusing on the undersides of"
          " the leaves where spores typically incubate. Avoid overhead"
          " watering to keep leaf surfaces dry."
      )
    else:
      st.info(
          "🟠 **Diagnosis: EARLY-STAGE SUB-CLINICAL STRESS** — Minor"
          " degradation starting to manifest.\n\n"
          "**Environmental Audit:** Check the surrounding microclimate—verify"
          " if the plant is receiving too much direct midday sunlight or"
          " experiencing root-bound stress in its current container.\n\n"
          "**Soil Check:** Test soil compaction and moisture retention; aerate"
          " the topsoil gently to improve oxygen flow to the root zone."
      )
