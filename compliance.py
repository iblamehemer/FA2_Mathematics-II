"""
Utility functions for PPE compliance analysis.

This module defines helper routines used by the Streamlit application to
post‑process detections from a YOLO model, assign personal protective equipment
(PPE) items to detected people, compute compliance status (GREEN/YELLOW/RED),
and draw annotated output images.

Functions:

    compute_iou(boxA, boxB) -> float
        Compute the Intersection over Union (IoU) between two bounding boxes.

    assign_ppe_to_people(dets, model_names) -> Tuple[List[dict], List[dict]]
        Separate person detections from PPE detections.  Returns two lists:
        a list of person dictionaries and a list of PPE dictionaries.

    compute_compliance_for_people(img_np, people, ppe, iou_thr, show_ppe_boxes,
                                  show_person_boxes, show_labels)
        Assign PPE items to each person, compute compliance status and
        return a list of per‑person summaries together with an annotated
        image (numpy array).

    draw_annotations(img_np, people, ppe, iou_thr, show_ppe_boxes,
                     show_person_boxes, show_labels)
        A convenience wrapper around compute_compliance_for_people that
        returns only the annotated image.

Constants:

    DEFAULT_PPE_CLASSES
        A list of PPE class names considered when determining compliance.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2

# Classes used for compliance evaluation.  We focus on three positive items
# (Hardhat, Mask, Safety Vest) and their corresponding negative classes.
DEFAULT_PPE_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    boxA, boxB : list of float
        Bounding boxes in [x1, y1, x2, y2] format.  Coordinates are
        expected to be absolute pixel positions.

    Returns
    -------
    float
        IoU value between 0 and 1.  Returns 0 if there is no overlap.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    boxA_area = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def assign_ppe_to_people(
    dets: List[Dict[str, Any]], model_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split raw detections into people and PPE items.

    The YOLO model outputs a list of detection dictionaries (see app.py) with
    bounding boxes, class IDs and class names.  This function separates
    person detections (class 'Person') from PPE detections.

    Parameters
    ----------
    dets : list of dict
        Each dictionary must have keys: 'xyxy' (list of floats), 'cls_id'
        (int), 'cls_name' (str) and 'conf' (float).
    model_names : dict
        Mapping from class ID to class name.  Not used here but kept for
        compatibility.

    Returns
    -------
    people : list of dict
        Each dictionary contains the bounding box and assigned index for a
        detected person (id and coordinates).
    ppe : list of dict
        Each dictionary contains the bounding box, class name, class id and
        confidence score for a PPE detection.
    """
    people: List[Dict[str, Any]] = []
    ppe: List[Dict[str, Any]] = []
    for det in dets:
        cls_name = det.get("cls_name", "")
        if cls_name.lower() == "person" or cls_name == "Person":
            person_dict = {
                "xyxy": det["xyxy"],
                "person_id": len(people) + 1,  # assign 1‑based IDs
            }
            people.append(person_dict)
        else:
            ppe.append({
                "xyxy": det["xyxy"],
                "cls_name": cls_name,
                "cls_id": det.get("cls_id"),
                "conf": det.get("conf", 0.0),
            })
    return people, ppe


def compute_compliance_for_people(
    img_np: np.ndarray,
    people: List[Dict[str, Any]],
    ppe: List[Dict[str, Any]],
    iou_thr: float = 0.25,
    show_ppe_boxes: bool = True,
    show_person_boxes: bool = True,
    show_labels: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Assign PPE items to each person, compute compliance and annotate image.

    Parameters
    ----------
    img_np : ndarray
        Original image array (BGR) on which annotations will be drawn.
    people : list of dict
        Each dictionary must contain keys 'xyxy' and 'person_id'.
    ppe : list of dict
        Each dictionary must contain keys 'xyxy', 'cls_name', 'cls_id', 'conf'.
    iou_thr : float, optional
        Minimum IoU required to associate a PPE detection with a person.
    show_ppe_boxes : bool, optional
        Whether to draw bounding boxes for individual PPE items.
    show_person_boxes : bool, optional
        Whether to draw bounding boxes around people.
    show_labels : bool, optional
        Whether to draw textual labels on the image.

    Returns
    -------
    people_status : list of dict
        For each person, a summary dictionary with keys:
        'person_id', 'status', 'present_items', 'missing_items', 'violations',
        'num_ppe_detected'.
    annotated : ndarray
        The annotated image with bounding boxes and labels drawn.
    """
    # Copy image to draw annotations
    annotated = img_np.copy()
    height, width = annotated.shape[:2]

    # Define required PPE items and mapping from class names to item types
    required = {"hardhat", "mask", "vest"}
    # Map class names (case insensitive) to simplified item identifiers
    name_to_item = {
        "hardhat": "hardhat",
        "no-hardhat": "hardhat",
        "mask": "mask",
        "no-mask": "mask",
        "safety vest": "vest",
        "no-safety vest": "vest",
    }

    # Initialize per‑person tracking structures
    person_data: List[Dict[str, Any]] = []
    for person in people:
        pdata = {
            "person_id": person["person_id"],
            "xyxy": person["xyxy"],
            "present": set(),
            "missing": set(),
            "num_ppe_detected": 0,
        }
        person_data.append(pdata)

    # Assign each PPE detection to the person with the highest IoU above threshold
    for ppe_det in ppe:
        best_iou = 0.0
        target_idx = None
        ppe_box = ppe_det["xyxy"]
        for idx, pdata in enumerate(person_data):
            iou_val = compute_iou(ppe_box, pdata["xyxy"])
            if iou_val > best_iou:
                best_iou = iou_val
                target_idx = idx
        if target_idx is not None and best_iou >= iou_thr:
            pdata = person_data[target_idx]
            # Determine item type (hardhat/mask/vest) from class name
            cls_name = ppe_det["cls_name"].lower()
            # Replace hyphen with space for matching
            cls_name_norm = cls_name.replace("-", " ")
            # Remove extra spaces
            cls_name_norm = " ".join(cls_name_norm.split())
            # Determine item key
            item_key = None
            for key, val in name_to_item.items():
                if key == cls_name_norm:
                    item_key = val
                    break
            if item_key:
                # Mark presence and missing based on positive or negative class
                if cls_name.startswith("no-"):
                    pdata["missing"].add(item_key)
                else:
                    pdata["present"].add(item_key)
                pdata["num_ppe_detected"] += 1

        # Draw PPE box if requested
        if show_ppe_boxes:
            x1, y1, x2, y2 = [int(v) for v in ppe_box]
            # Blue color for PPE boxes (BGR)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if show_labels:
                label = ppe_det["cls_name"]
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    # Compute final status for each person and draw annotations
    people_status: List[Dict[str, Any]] = []
    for pdata in person_data:
        missing = required - pdata["present"]
        # If negative detections exist, treat those items as missing
        missing |= pdata["missing"]
        # Determine status
        if not missing:
            status = "GREEN"
            color = (0, 255, 0)  # Green
        elif pdata["present"]:
            status = "YELLOW"
            color = (0, 255, 255)  # Yellow (BGR)
        else:
            status = "RED"
            color = (0, 0, 255)  # Red
        # Draw person box if requested
        if show_person_boxes:
            x1, y1, x2, y2 = [int(v) for v in pdata["xyxy"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                cv2.putText(
                    annotated,
                    f"{status}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        # Build status dictionary
        people_status.append({
            "person_id": pdata["person_id"],
            "status": status,
            "present_items": sorted(list(pdata["present"])) if pdata["present"] else [],
            "missing_items": sorted(list(missing)) if missing else [],
            "violations": len(missing),
            "num_ppe_detected": pdata["num_ppe_detected"],
        })

    return people_status, annotated


def draw_annotations(
    img_np: np.ndarray,
    people: List[Dict[str, Any]],
    ppe: List[Dict[str, Any]],
    iou_thr: float = 0.25,
    show_ppe_boxes: bool = True,
    show_person_boxes: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw annotated boxes and return the image.

    This convenience wrapper calls compute_compliance_for_people and returns
    only the annotated image.  It can be used by callers that do not need
    the per‑person status list.
    """
    _, annotated = compute_compliance_for_people(
        img_np,
        people,
        ppe,
        iou_thr=iou_thr,
        show_ppe_boxes=show_ppe_boxes,
        show_person_boxes=show_person_boxes,
        show_labels=show_labels,
    )
    return annotated