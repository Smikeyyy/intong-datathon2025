# Import necessary libraries
import cv2
import numpy as np
import pandas as pd
import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

# Define brand multipliers, severity multipliers, panel multipliers, and damage lookup
brand_multipliers = {
"Wuling": 1.00,
"BYD": 1.05,
"Hyundai": 1.10,
"Nissan": 1.15,
"Toyota": 1.20,
"Honda": 1.20,
"Volkswagen": 1.25,
"BMW": 1.60,
"Audi": 1.60,
"Mercedes Benz": 1.70
}

damage_lookup = {
"crack":1400000,
"dent":700000,
"scratch":350000,
"glass shatter":2000000,
"lamp broken":1500000,
"tire flat":500000
}

severity_multipliers = {
"minor":0.33,
"moderate":0.67,
"major":1.00
}

panel_multipliers = {
"back_bumper":1.00,
"back_door":0.80,
"back_glass":0.30,
"back_left_door":0.80,
"back_left_light":0.20,
"back_light":0.20,
"back_right_door":0.80,
"back_right_light":0.20,
"front_bumper":1.00,
"front_door":0.80,
"front_glass":0.30,
"front_left_door":0.80,
"front_left_light":0.20,
"front_light":0.20,
"front_right_door":0.80,
"front_right_light":0.20,
"hood":1.00,
"left_mirror":0.50,
"object":0.10,
"right_mirror":0.50,
"tailgate":1.20,
"trunk":0.40,
"wheel":0.25
}

# Define bumper cost
bumper_cost = 1000000

# Define both car parts segmentation and damage type segmentation model
model_cps = YOLO("best_cps.pt")
model_dts = YOLO("best_dts.pt")

# Function to segment car parts and damage type
def predict_image(brand, img, part_conf_threshold, part_iou_threshold, damage_conf_threshold, damage_iou_threshold):
    
    # Define total cost
    total_cost = 0

    # Define dataframe for report
    df = pd.DataFrame(columns = ["Part", "Type", "Severity", "Cost"])

    # Performs Car Parts Segmentation
    results_cps = model_cps.predict(
        source = img,
        conf = part_conf_threshold,
        iou = part_iou_threshold,
        show_labels = True,
        show_conf = True,
        imgsz = 640,
    )

    # Get panel segmentation masks
    panel_masks = results_cps[0].masks.data

    # Iterate through all results
    for r in results_cps:
        im_array = r.plot()
        im_cps = Image.fromarray(im_array[..., ::-1])        

    # Get car parts id and map back to class names
    id2class_cps = results_cps[0].names
    predicted_class_ids_cps = list(results_cps[0].boxes.cls)
    predicted_class_names_cps = [id2class_cps[int(class_id)] for class_id in predicted_class_ids_cps]

    # # Version 1: Cropping Detected Parts

    # # Get all boxes and crop as images
    # boxes = results_cps[0].boxes.xyxy.cpu().numpy()

    # # Iterate through all detected parts
    # crops = []
    # for box in boxes:
    #     xmin, ymin, xmax, ymax = map(int, box)
    #     cropped = img.crop((xmin, ymin, xmax, ymax))
    #     crops.append(cropped)

    # Version 2: Crop Detected Parts from Original Image

    # Load orinal image for cropping
    image = results_cps[0].orig_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height = image.shape[1], image.shape[0]

    # Iterate through each masks and get segmented image
    crops = []
    for i in range(len(panel_masks)):
        mask_image = panel_masks[i].cpu().numpy()
        mask_image = cv2.resize(mask_image, (width, height), interpolation = cv2.INTER_NEAREST)
        mask_image = mask_image.astype(np.uint8)
        mask_image = image * mask_image[:, :, np.newaxis]
        crops.append(mask_image)

    # For each image in crops perform damage type segmentation
    # Here batching is not performed since our device doesn't have a GPU
    damages = []
    for panel_mask, im_dts, part_name in zip(panel_masks, crops, predicted_class_names_cps):
        # Performs Damage Type Segmentation
        results_dts = model_dts.predict(
            source = im_dts,
            conf = damage_conf_threshold,
            iou = damage_iou_threshold,
            show_labels = True,
            show_conf = True,
            imgsz = 640,
        )

        # Check if any damage is detected and if not move to next part
        if not results_dts[0].boxes:
            print(f"No damage in part {part_name}. Skipping.")
            continue

        # Iterate through all results
        for r in results_dts:
            im_array = r.plot()
            im_dts = Image.fromarray(im_array[..., ::-1])
            
            # Save im_dts in damages
            damages.append(im_dts)

        # Define severity from masks
        severities = []
        damage_masks = results_dts[0].masks.data
        for i in range(len(damage_masks)):
            mask_image = damage_masks[i].cpu().numpy()
            area = cv2.countNonZero(mask_image)
            total_area = cv2.countNonZero(panel_mask.cpu().numpy())
            iou = area/total_area
            if iou < 0.05:
                severity = "minor"
                severities.append(severity)
            elif iou < 0.20:
                severity = "moderate"
                severities.append(severity)
            else:
                severity = "major"
                severities.append(severity)
        
        # Check brand name and get corresponding multiplier
        if brand == "Wuling":
            multiplier = brand_multipliers["Wuling"]
        elif brand == "BYD":
            multiplier = brand_multipliers["BYD"]
        elif brand == "Hyundai":
            multiplier = brand_multipliers["Hyundai"]
        elif brand == "Nissan":
            multiplier = brand_multipliers["Nissan"]
        elif brand == "Toyota":
            multiplier = brand_multipliers["Toyota"]
        elif brand == "Honda":
            multiplier = brand_multipliers["Honda"]
        elif brand == "Volkswagen":
            multiplier = brand_multipliers["Volkswagen"]
        elif brand == "BMW":
            multiplier = brand_multipliers["BMW"]
        elif brand == "Audi":
            multiplier = brand_multipliers["Audi"]
        elif brand == "Mercedes Benz":
            multiplier = brand_multipliers["Mercedes Benz"]

        # Get damage id and map back to class names
        id2class_dts = results_dts[0].names
        predicted_class_ids_dts = list(results_dts[0].boxes.cls)
        predicted_class_names_dts = [id2class_dts[int(class_id)] for class_id in predicted_class_ids_dts]

        # Estimate total cost by brand, part, severity, and damage types
        for class_name, severity in zip(predicted_class_names_dts, severities):
            if class_name in ["glass shatter", "lamp broken", "tire flat"]:
                cost = multiplier * bumper_cost * panel_multipliers[part_name]
                total_cost += cost

                # Add instance to dataframe
                new_entry = {
                "Part": part_name,
                "Type": class_name,
                "Severity": severity,
                "Cost": f"Rp {int(cost/10) * 10:,}".replace(",", ".")
                }

                # Add the new row at the next index
                df.loc[len(df)] = new_entry

            else:
                if severity == "Major":
                    cost = multiplier * bumper_cost * panel_multipliers[part_name]
                    total_cost += cost

                    # Add instance to dataframe
                    new_entry = {
                    "Part": part_name,
                    "Type": class_name,
                    "Severity": severity,
                    "Cost": f"Rp {int(cost/10) * 10:,}".replace(",", ".")
                    }

                    # Add the new row at the next index
                    df.loc[len(df)] = new_entry

                else:
                    cost = multiplier * severity_multipliers[severity] * panel_multipliers[part_name] * damage_lookup[class_name]
                    total_cost += cost

                    # Add instance to dataframe
                    new_entry = {
                    "Part": part_name,
                    "Type": class_name,
                    "Severity": severity,
                    "Cost": f"Rp {int(cost/10) * 10:,}".replace(",", ".")
                    }

                    # Add the new row at the next index
                    df.loc[len(df)] = new_entry

    total_cost = f"Rp {int(total_cost/10) * 10:,}".replace(",", ".")

    return im_cps, damages, df, total_cost

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Dropdown(["Toyota", "Honda", "Nissan", "BYD", "Wuling", 
                     "Hyundai", "Mercedes Benz", "Audi", "Volkswagen", "BMW"], 
                     label = "Brand"),
        gr.Image(type = "pil", label = "Upload Image"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.50, label = "Part - Confidence threshold"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.50, label = "Part - IoU threshold"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.50, label = "Damage - Confidence threshold"),
        gr.Slider(minimum = 0, maximum = 1, value = 0.50, label = "Damage - IoU threshold")
    ],
    outputs = [gr.Image(type = "pil", label = "Car Part Segmentation"),
               gr.Gallery(label = "Car Damage Segmentation"),
               gr.Dataframe(headers = ["Part", "Type", "Severity", "Cost"], label = ""),
               gr.Label(label = "Estimated Total Cost")],
    title = "Car Damage Detection using YOLOv11 - Nano",
    theme = gr.themes.Soft(),
)

if __name__ == "__main__":
    iface.launch()