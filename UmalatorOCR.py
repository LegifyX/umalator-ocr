import base64
import csv
import gzip
import http.server
import hashlib
import json
import locale
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import BooleanVar, messagebox, ttk
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageGrab
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from rapidocr_onnxruntime import RapidOCR
from aptitude_extractor import AptitudeExtractor

#Configuration and Constants
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)
UMALATOR_BASE_URL = "https://alpha123.github.io/uma-tools/umalator-global/"
REPO_URL_TOOLS = "https://github.com/alpha123/uma-tools"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("UmalatorOCR")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_CSV_PATH = DATA_DIR / "runners.csv"
EXTERNAL_DIR = BASE_DIR / "external"
TOOLS_DIR = EXTERNAL_DIR / "uma-tools"

# OCR settings
OCR_NAME_Y_MAX = 400
OCR_NAME_X_MIN_FRAC = 0.5
OCR_SKILLS_Y_MIN_FRAC = 0.25
OCR_FUZZY_MATCH_CUTOFF = 80
OCR_APTITUDE_Y_MIN_FRAC = 0.295  # Y-start for aptitude block
OCR_APTITUDE_Y_MAX_FRAC = 0.420  # Y-end for aptitude block

# Defaults
DEFAULT_COURSE_ID = 10606
DEFAULT_NSAMPLES = 500
DEFAULT_USE_POS_KEEP = True
DEFAULT_RACEDEF = {"mood": 2, "ground": 1, "weather": 1, "season": 1, "time": 2, "grade": 100}

# GUI colors
COLOR_DEFAULT_BG = "white"
COLOR_SELECTED_BG = "#cce5ff"
COLOR_DUP_BG = "#fff8c6"
COLOR_STATUS_BAR_BG = "#e0e0e0"
COLOR_STATUS_BAR_FG = "#333333"

# EN > JP names for strategy (Internal mapping for GUI)
STRATEGY_DISPLAY_TO_INTERNAL = {
    "Front Runner": "Nige",
    "Pace Chaser": "Senkou",
    "Late Surger": "Sashi",
    "End Closer": "Oikomi",
}

# Strategy mapping and ranking (tiebreak order)
STRATEGY_INTERNAL_TO_DISPLAY = {v: k for k, v in STRATEGY_DISPLAY_TO_INTERNAL.items()}
STRATEGY_OCR_TO_INTERNAL = {"Front": "Nige", "Pace": "Senkou", "Late": "Sashi", "End": "Oikomi"}
STRATEGY_TIEBREAK_ORDER = ["Nige", "Senkou", "Sashi", "Oikomi"]

# Aptitude ranking (S > G scale)
APTITUDES = ["S", "A", "B", "C", "D", "E", "F", "G"]
APTITUDE_RANK = {apt: i for i, apt in enumerate(reversed(APTITUDES))}  # G=0, S=7

# Race condition choices and mappings
GROUND_CONDITIONS = ["Firm", "Good", "Soft", "Heavy"]
GROUND_MAP = {name: i + 1 for i, name in enumerate(GROUND_CONDITIONS)}
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy"]
WEATHER_MAP = {name: i + 1 for i, name in enumerate(WEATHER_CONDITIONS)}
SEASON_CONDITIONS = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_MAP = {name: i + 1 for i, name in enumerate(SEASON_CONDITIONS)}


# Data classes
@dataclass
class Horse:
    name: str = ""
    speed: int = 0
    stamina: int = 0
    power: int = 0
    guts: int = 0
    wisdom: int = 0
    skills: List[str] = field(default_factory=list)
    outfitId: str = ""
    strategy: str = "Senkou"
    distanceAptitude: str = "S"
    surfaceAptitude: str = "A"
    strategyAptitude: str = "A"

    def to_json(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# Utility functions
def ensure_repo(url: str, path: Path):
    """Clone repository if missing (quiet)."""
    if path.exists():
        return
    logger.info("Cloning %s -> %s", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    except Exception as e:
        logger.warning("Failed to clone repo: %s", e)
        raise


# Data Manager
class DataManager:
    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self.skill_names: List[str] = []
        self.uma_names: List[str] = []
        self.epithet_names: List[str] = []
        self.skill_norm_map: Dict[str, str] = {}
        self.uma_norm_map: Dict[str, str] = {}
        self.epithet_norm_map: Dict[str, str] = {}
        self.skill_id_map: Dict[str, Dict[str, str]] = {}
        self.uma_id_map: Dict[str, str] = {}
        self.uma_outfit_map: Dict[str, Dict[str, str]] = {}
        # course/track data
        self.course_data: Dict[str, dict] = {}
        self.track_names: Dict[str, List[str]] = {}
        self.track_index: Dict[str, List[dict]] = {}

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9◎○]", "", (s or "").lower())

    def _ensure_repo(self):
        ensure_repo(REPO_URL_TOOLS, self.tools_dir)

    def load_all_data(self):
        logger.info("Loading data from uma-tools...")
        self._ensure_repo()
        self._load_skills()
        self._load_umas()
        self.load_course_data()
        logger.info("Data load complete.")

    def _load_skills(self):
        try:
            fpath = self.tools_dir / "umalator-global" / "skillnames.json"
            with open(fpath, encoding="utf-8") as f:
                skill_data = json.load(f)
        except Exception as e:
            logger.error("Failed to load skillnames.json: %s", e)
            return

        self.skill_names = [names[0] for names in skill_data.values() if names]
        self.skill_norm_map = {self._normalize(name): name for name in self.skill_names}
        self.skill_id_map = {}
        for skill_id, names in skill_data.items():
            for name in names:
                key = name.lower()
                entry = self.skill_id_map.setdefault(key, {"normal": "", "inherited": ""})
                if skill_id.startswith("9"):
                    if not entry["inherited"]:
                        entry["inherited"] = skill_id
                else:
                    if not entry["normal"]:
                        entry["normal"] = skill_id

    def _load_umas(self):
        try:
            fpath = self.tools_dir / "umalator-global" / "umas.json"
            with open(fpath, encoding="utf-8") as f:
                uma_data = json.load(f)
        except Exception as e:
            logger.error("Failed to load umas.json: %s", e)
            return

        for v in uma_data.values():
            names = [n for n in v.get("name", []) if n]
            if not names:
                continue
            self.uma_names.extend(names)
            canonical = names[-1]
            self.uma_id_map[self._normalize(canonical)] = canonical
            outfit_dict = {self._normalize(epithet): outfit_id for outfit_id, epithet in v.get("outfits", {}).items()}
            self.uma_outfit_map[canonical] = outfit_dict
            self.epithet_names.extend([ep for ep in v.get("outfits", {}).values() if ep])
        self.uma_norm_map = {self._normalize(n): n for n in self.uma_names}
        self.epithet_norm_map = {self._normalize(e): e for e in self.epithet_names}

    def load_course_data(self):
        """Loads course_data.json and tracknames.json with robust parsing."""
        course_path = self.tools_dir / "umalator-global" / "course_data.json"
        track_path = self.tools_dir / "umalator-global" / "tracknames.json"

        try:
            with open(course_path, encoding="utf-8") as f:
                self.course_data = json.load(f)
        except Exception as e:
            logger.error("Failed to load course_data.json: %s", e)
            self.course_data = {}

        try:
            with open(track_path, encoding="utf-8") as f:
                self.track_names = json.load(f)
        except Exception as e:
            logger.error("Failed to load tracknames.json: %s", e)
            self.track_names = {}

        self.track_index = {}
        for cid, item in self.course_data.items():
            if isinstance(item, dict):
                rt = item.get("raceTrackId")
                if rt is not None:
                    self.track_index.setdefault(str(rt), []).append(item)


# OCR Processor
class OCRProcessor:
    _CIRCLE_ALIASES = {"o": "○", "O": "○", "0": "○", "〇": "○", "◎": "◎", "○": "○"}
    STRATEGY_KEYWORDS = {"Front": "Nige", "Pace": "Senkou", "Late": "Sashi", "End": "Oikomi"}
    APTITUDE_ORDER = {"S": 7, "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self._ocr = None  # Initialize the attribute as None
        self.aptitude_extractor = AptitudeExtractor()  # Add improved extractor
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("OCRProcessor")

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            logger.info("Initializing RapidOCR engine...")
            self._ocr = RapidOCR()
        return self._ocr

    def extract_data_from_image(self, image_source: Union[str, np.ndarray], use_text_matching: bool = False) -> Dict[str, str]:
        """Extract data from an image and return a dictionary of attributes."""
        try:
            logger.debug("Processing image source: %s", image_source)
            if isinstance(image_source, str):
                img = cv2.imread(image_source)
                if img is None:
                    raise IOError(f"Cannot read image {image_source}. Ensure the file exists and is accessible.")
            else:
                img = image_source
                if not isinstance(img, np.ndarray):
                    raise ValueError("Invalid image data provided")

            h, w, _ = img.shape
            logger.debug("Image dimensions: height=%d, width=%d", h, w)

            # Preprocess image
            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #enhanced_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Perform OCR
            res, _ = self.ocr(image_source)
            logger.debug("OCR Results:")

            # --- Force dump OCR results every time ---
            import json, os

            debug_dir = "logs"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "ocr_debug.json")

            normalized = []
            try:
                if res:
                    for box, text, conf in res:
                        try:
                            box_list = box.tolist() if hasattr(box, "tolist") else list(box)
                        except Exception:
                            box_list = str(box)  # fallback if it's something weird
                        normalized.append({
                            "text": str(text),
                            "conf": float(conf) if conf is not None else None,
                            "box": box_list,
                        })
                else:
                    logger.warning("OCR returned no results for %s", image_source)

                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(normalized, f, ensure_ascii=False, indent=2)

                logger.debug("Wrote %d OCR results to %s", len(normalized), debug_path)
            except Exception as dump_err:
                logger.error("Failed to dump OCR results: %s", dump_err)
                # Write at least a placeholder file
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write("[]")

            # Call the debug method to draw rectangles and save debug image
            self.debug_aptitude_block(image=img, ocr_results=res,
                                      img_height=h)  # This line ensures debug image is created.

        except Exception as e:
            logger.error("OCR failed: %s", e)
            return {}

        stats = {}
        name, epithet = self._extract_identity(res, w, h)
        stats["Name"] = name
        stats["Epithet"] = epithet

        stat_values = self._extract_stats(res)
        stat_names = ["Speed", "Stamina", "Power", "Guts", "Wit"]
        stats.update(zip(stat_names, stat_values))

        # Pass use_text_matching=True to toggle text-based matching
        aptitudes = AptitudeExtractor.extract_from_ocr_results(res)
        stats.update(AptitudeExtractor.extract_from_ocr_results(res))
        stats["Skills"] = "|".join(self._extract_skills(res, img))
        return stats

    # debug aptitude block - remove when done
    def debug_aptitude_block(self, image: np.ndarray, ocr_results: List, img_height: int) -> None:
        # Clone the input image to draw on
        debug_image = image.copy()

        # Define Y-coordinate boundaries for each row based on image height
        y_min_surface = int(img_height * 0.305)
        y_max_surface = int(img_height * 0.340)

        y_min_distance = int(img_height * 0.340)
        y_max_distance = int(img_height * 0.370)

        y_min_style = int(img_height * 0.370)
        y_max_style = int(img_height * 0.405)

        # Draw the boundaries - currently surface=green, distance=blue, style=red
        rows = [
            ("Surface", (y_min_surface, y_max_surface), (0, 255, 0)),
            ("Distance", (y_min_distance, y_max_distance), (255, 0, 0)),
            ("Style", (y_min_style, y_max_style), (0, 0, 255)),
        ]

        for row_name, (y_min, y_max), color in rows:
            cv2.rectangle(
                debug_image,
                (0, y_min),
                (image.shape[1], y_max),
                color,
                2,
            )
            self.logger.debug(f"{row_name} Row: Y Range = {y_min} to {y_max}")

        # What the OCR sees
        for box, text, _ in ocr_results:
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            x_max = max([p[0] for p in box])
            y_max = max([p[1] for p in box])

            # OCR box
            cv2.rectangle(debug_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 1)
            # OCR text
            cv2.putText(
                debug_image,
                text,
                (int(x_min), int(y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # save debug image in root dir
        debug_path = "debug_aptitude_block.png"
        cv2.imwrite(debug_path, debug_image)
        self.logger.info(f"Debug image saved to {debug_path}")

    # extracting uma name and epithet
    def _extract_identity(self, ocr_results: List, img_width: int, img_height: int) -> Tuple[str, str]:
        best_score = 0
        chosen = "Unknown"
        name_box_top = img_height
        for box, text, _ in ocr_results:
            y1 = max(p[1] for p in box)
            x0 = min(p[0] for p in box)
            if y1 < OCR_NAME_Y_MAX and x0 > img_width * OCR_NAME_X_MIN_FRAC:
                norm = self.data_manager._normalize(text)
                if not norm:
                    continue
                match = process.extractOne(norm, self.data_manager.uma_norm_map.keys(), scorer=fuzz.ratio)
                if match and match[1] > best_score:
                    best_score = match[1]
                    chosen = self.data_manager.uma_norm_map[match[0]]
                    name_box_top = min(p[1] for p in box)

        epithet = ""
        if chosen != "Unknown":
            candidates = []
            for box, text, _ in ocr_results:
                y1 = max(p[1] for p in box)
                x0 = min(p[0] for p in box)
                if y1 <= name_box_top and x0 > img_width * OCR_NAME_X_MIN_FRAC:
                    candidates.append((min(p[1] for p in box), text))
            candidates.sort()
            joined = self.data_manager._normalize(" ".join(t for _, t in candidates))
            if joined:
                match = process.extractOne(joined, self.data_manager.epithet_norm_map.keys(), scorer=fuzz.ratio)
                if match:
                    epithet = self.data_manager.epithet_norm_map[match[0]]
        return chosen, epithet

    # extracting stats
    def _extract_stats(self, ocr_results: List) -> List[str]:
        """
        Extract stats (Speed, Stamina, Power, Guts, Wit) by pairing label positions
        with the nearest numeric OCR candidate. Falls back to row grouping.
        """
        labels = ["Speed", "Stamina", "Power", "Guts", "Wit"]
        label_positions = {}
        number_candidates = []

        # collect number candidates and label positions
        for box, text, _ in ocr_results:
            t = text.strip()
            # numbers (2-4 digits)
            if re.fullmatch(r"\d{2,4}", t):
                try:
                    x_center = sum(p[0] for p in box) / len(box)
                    y_center = sum(p[1] for p in box) / len(box)
                    number_candidates.append((x_center, y_center, int(t)))
                except Exception:
                    continue
            # labels
            if t in labels:
                x_center = sum(p[0] for p in box) / len(box)
                y_center = sum(p[1] for p in box) / len(box)
                label_positions[t] = (x_center, y_center)

        # if we have labels and numbers, pair each label to nearest number
        if label_positions and number_candidates:
            result = []
            for lab in labels:
                if lab not in label_positions:
                    result.append("")  # missing label
                    continue
                lx, ly = label_positions[lab]
                # choose nearest number by Euclidean distance
                nearest = min(number_candidates, key=lambda n: math.hypot(n[0] - lx, n[1] - ly))
                result.append(str(nearest[2]))
            return result

        # fallback to previous row-grouping approach if labels aren't found
        # build numeric list and group by Y rows
        candidates = []
        for box, text, _ in ocr_results:
            if re.fullmatch(r"\d{2,4}", text.strip()):
                try:
                    y_center = sum(p[1] for p in box) / len(box)
                    x_center = sum(p[0] for p in box) / len(box)
                    candidates.append((x_center, y_center, int(text)))
                except Exception:
                    continue

        grouped_rows = OCRProcessor._group_stats_by_rows(candidates)
        if grouped_rows:
            stats_row = grouped_rows[0]
            return [str(num) for _, _, num in stats_row]

        return ["", "", "", "", ""]

    def _group_stats_by_rows(number_candidates: List[Tuple[float, float, int]]) -> List[List[Tuple[float, float, int]]]:
        if not number_candidates:
            return []

        # Sort candidates by vertical position (y value)
        number_candidates.sort(key=lambda x: x[1])  # Sort by Y position

        grouped_rows = []
        current_row = []
        current_y = number_candidates[0][1]

        for x, y, num in number_candidates:
            # Check if the current number is close enough to the previous row
            if abs(y - current_y) > 20:  # Threshold for a new row
                if current_row:
                    # Sort row by horizontal position (x value) and add to grouped_rows
                    grouped_rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [(x, y, num)]
                current_y = y
            else:
                current_row.append((x, y, num))

        # Add the last row
        if current_row:
            grouped_rows.append(sorted(current_row, key=lambda x: x[0]))

        return grouped_rows

    def _group_by_column(self, lines: List) -> List[str]:
        if not lines: return []
        lines.sort(key=lambda l: l[1])
        grouped = []
        for x0, y0, x1, y1, text in lines:
            if grouped and y0 - grouped[-1][3] < 25:
                grouped[-1][4].append(text)
                grouped[-1][2] = max(grouped[-1][2], x1)
                grouped[-1][3] = max(grouped[-1][3], y1)
            else:
                grouped.append([x0, y0, x1, y1, [text]])
        return [" ".join(g[4]) for g in grouped]

    def _group_skill_boxes(self, ocr_results: List, img_height: int) -> List[Tuple[int, int, int, int, str]]:
        """
        Group OCR text boxes into left/right columns for skills and return
        a list of tuples (x0, y0, x1, y1, text) for each grouped box.
        """
        # Gather candidate boxes that are in the skill area (dynamic threshold)
        lines = []
        skills_threshold = img_height * OCR_SKILLS_Y_MIN_FRAC
        for box, text, _ in ocr_results:
            x0 = int(min(p[0] for p in box))
            y0 = int(min(p[1] for p in box))
            x1 = int(max(p[0] for p in box))
            y1 = int(max(p[1] for p in box))
            # only consider boxes that are below the skill threshold
            if y0 < skills_threshold:
                continue
            lines.append([x0, y0, x1, y1, text])

        if not lines:
            return []

        # split by approximate midline to left/right columns
        xs = [l[0] for l in lines]
        mid = (min(xs) + max(xs)) / 2.0
        left = [l for l in lines if l[0] < mid]
        right = [l for l in lines if l[0] >= mid]

        def group_column_boxes(col_lines):
            col_lines.sort(key=lambda l: l[1])  # sort by y
            grouped = []
            for x0, y0, x1, y1, txt in col_lines:
                if grouped and y0 - grouped[-1][3] < 25:
                    # same row: merge
                    grouped[-1][2] = max(grouped[-1][2], x1)
                    grouped[-1][3] = max(grouped[-1][3], y1)
                    grouped[-1][4].append(txt)
                else:
                    grouped.append([x0, y0, x1, y1, [txt]])
            # flatten text lists
            return [(g[0], g[1], g[2], g[3], " ".join(g[4])) for g in grouped]

        left_g = group_column_boxes(left)
        right_g = group_column_boxes(right)

        # Combine the groups: emulate interleaving left/right reading order.
        groups = []
        # zip_longest keeps order but we want all left groups then all right groups interleaved
        # Simpler: maintain the order left followed by right, as your original code did.
        groups.extend(left_g)
        groups.extend(right_g)
        return groups

    def _detect_circle(self, img: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Given the image and a bounding rect (x0,y0,x1,y1), crop a small area to the right
        of the rect and try to detect a circle glyph using HoughCircles. Returns '◎' or '○' or None.
        """
        try:
            x0, y0, x1, y1 = map(int, rect)
            margin = 5
            # crop slightly to the right of the text box (common place for circle)
            xs = max(0, x1 - 50)
            xe = min(img.shape[1], x1 + 30)
            ys = max(0, y0 - margin)
            ye = min(img.shape[0], y1 + margin)
            crop = img[ys:ye, xs:xe]
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=15, minRadius=5, maxRadius=30
            )
            if circles is None:
                return None
            # choose the largest circle candidate
            cx, cy, r = max(circles[0], key=lambda c: c[2])
            canny = cv2.Canny(gray, 50, 150)
            mask = np.zeros_like(canny)
            inner_r = int(r * 0.55) or 1
            cv2.circle(mask, (int(cx), int(cy)), inner_r, 255, 2)
            count = cv2.countNonZero(cv2.bitwise_and(canny, mask))
            coverage = count / (2 * math.pi * inner_r) if inner_r > 0 else 0.0
            # threshold similar to older script
            if coverage > 0.6:
                return "◎"
            return "○"
        except Exception:
            return None

    def _extract_skills(self, ocr_results: List, image: Optional[np.ndarray] = None) -> List[str]:
        """
        Robust skill extraction:
          - group OCR boxes into left/right columns
          - detect circle glyphs visually if OCR didn't include them
          - normalize + fuzzy-match to canonical skill names
        """
        if image is None:
            # fallback: use prior naive grouping (existing method) but still normalize circles
            grouped_texts = self._group_by_column([
                (min(p[0] for p in box), min(p[1] for p in box),
                 max(p[0] for p in box), max(p[1] for p in box), text)
                for box, text, _ in ocr_results
            ])
            groups = [(0, 0, 0, 0, t) for t in grouped_texts]
        else:
            groups = self._group_skill_boxes(ocr_results, image.shape[0])

        extracted_skills = []
        seen = set()
        for x0, y0, x1, y1, text in groups:
            # attempt circle detection on the grouped box area if we have the image
            if image is not None:
                circle = self._detect_circle(image, (x0, y0, x1, y1))
                if circle and circle not in text:
                    text = f"{text} {circle}"

            # normalize circles & text then fuzzy-match
            # convert ascii lookalikes to desired circle glyphs first
            replaced = "".join(self._CIRCLE_ALIASES.get(ch, ch) for ch in text)
            norm = self.data_manager._normalize(replaced)
            if not norm:
                continue
            match = process.extractOne(
                norm, self.data_manager.skill_norm_map.keys(),
                scorer=fuzz.ratio, score_cutoff=OCR_FUZZY_MATCH_CUTOFF
            )
            if match:
                canonical_skill = self.data_manager.skill_norm_map[match[0]]
                if canonical_skill not in seen:
                    seen.add(canonical_skill)
                    extracted_skills.append(canonical_skill)

        return extracted_skills

# --- URL Generator ---
class UmalatorURLGenerator:
    def __init__(self, data_manager: DataManager):
        self.data = data_manager

    def create_url_from_rows(
            self, rows: List[Dict[str, str]], course_id: int, ground: int, weather: int, season: int,
    ) -> str:
        if len(rows) < 2: raise ValueError("Need two rows")

        uma1 = self._parse_horse_from_row(rows[0])
        uma2 = self._parse_horse_from_row(rows[1])

        racedef = DEFAULT_RACEDEF.copy()
        racedef.update({"ground": ground, "weather": weather, "season": season})
        payload = {"courseId": course_id, "nsamples": DEFAULT_NSAMPLES, "usePosKeep": DEFAULT_USE_POS_KEEP,
                   "racedef": racedef, "uma1": uma1.to_json(), "uma2": uma2.to_json()}

        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        zipped = gzip.compress(raw)
        return urllib.parse.quote(base64.b64encode(zipped).decode("ascii"))

    def _parse_horse_from_row(self, row: Dict[str, str]) -> Horse:
        skill_names = [n.strip().lower() for n in row.get("Skills", "").split("|") if n.strip()]
        skill_ids = []
        for name in skill_names:
            ids = self.data.skill_id_map.get(name)
            if ids and (skill_id := ids.get("normal") or ids.get("inherited")):
                skill_ids.append(skill_id)

        canonical_name, outfit_id = "", ""
        if name_input := row.get("Name", ""):
            key = self.data._normalize(name_input)
            if match := process.extractOne(key, self.data.uma_id_map.keys(), scorer=fuzz.ratio):
                canonical_name = self.data.uma_id_map[match[0]]

        if canonical_name and (epithet_input := row.get("Epithet", "")):
            if ep_map := self.data.uma_outfit_map.get(canonical_name, {}):
                key = self.data._normalize(epithet_input)
                if match := process.extractOne(key, ep_map.keys(), scorer=fuzz.ratio):
                    outfit_id = ep_map[match[0]]

        return Horse(
            name=canonical_name, outfitId=outfit_id,
            speed=int(row.get("Speed") or 0), stamina=int(row.get("Stamina") or 0),
            power=int(row.get("Power") or 0), guts=int(row.get("Guts") or 0),
            wisdom=int(row.get("Wit") or 0), skills=skill_ids,
            strategy=row.get("strategy", "Senkou"),
            distanceAptitude=row.get("distanceAptitude", "A"),
            surfaceAptitude=row.get("surfaceAptitude", "A"),
            strategyAptitude=row.get("strategyAptitude", "A"))


# --- File handling utils ---
CSV_FIELDS = ["Name", "Epithet", "Speed", "Stamina", "Power", "Guts", "Wit", "Skills",
              "surfaceAptitude", "distanceAptitude", "strategyAptitude", "strategy"]


def append_to_csv(row: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    full_row = {field: row.get(field, "") for field in CSV_FIELDS}
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header: writer.writeheader()
        writer.writerow(full_row)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Loads CSV rows with robust encoding fallback."""
    if not path.exists(): return []
    try:
        # First, try the standard 'utf-8-sig' which handles BOMs correctly.
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        # If that fails, it's likely a legacy system encoding (like cp1252 on Windows).
        logger.warning(f"UTF-8 decoding failed for {path}. Falling back to system default encoding.")
        try:
            fallback_encoding = locale.getpreferredencoding(False)
            with open(path, newline="", encoding=fallback_encoding) as f:
                return list(csv.DictReader(f))
        except Exception as e:
            logger.error(f"Failed to read CSV with fallback encoding '{fallback_encoding}': {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to load CSV {path}: {e}")
        return []


# --- Clipboard monitor ---
def monitor_clipboard(ocr_processor: OCRProcessor, callback: Optional[callable] = None) -> bool:
    try:
        img_pil = ImageGrab.grabclipboard()
        if not isinstance(img_pil, Image.Image):
            logger.debug("Clipboard does not contain a valid image")
            return False

        current_hash = hashlib.md5(img_pil.tobytes()).hexdigest()
        if getattr(monitor_clipboard, "last_hash", None) == current_hash:
            return False
        monitor_clipboard.last_hash = current_hash

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        if img_cv is None or not isinstance(img_cv, np.ndarray):
            raise ValueError("Failed to convert clipboard image to NumPy array")

        row = ocr_processor.extract_data_from_image(img_cv, use_text_matching=True)
        if row and row.get("Name") and row.get("Name") != "Unknown":
            append_to_csv(row, OUTPUT_CSV_PATH)
            PROCESSED_DIR.mkdir(exist_ok=True)
            img_pil.save(PROCESSED_DIR / f"clip_{int(time.time())}.png")
            if callback:
                callback(f"Added {row.get('Name', 'Unknown')} to runners.csv")
            return True
    except Exception as e:
        logger.error("Clipboard monitor failed: %s", e)
    return False


# --- Local server ---
def start_local_server() -> Tuple[http.server.ThreadingHTTPServer, threading.Thread, int]:
    """Starts a local server to serve the Umalator tool."""
    serve_dir = TOOLS_DIR / "umalator-global"

    class UmaToolsHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def translate_path(self, path: str) -> str:
            path = urllib.parse.urlparse(path).path
            if path.startswith("/uma-tools/"):
                return str(TOOLS_DIR / path[len("/uma-tools/"):])

            first_part = path.lstrip("/").split("/", 1)[0]
            if first_part in {"icons", "courseimages", "fonts", "strings", "skill_meta.json", "umas.json",
                              "icons.json"}:
                return str(TOOLS_DIR / path.lstrip("/"))

            return super().translate_path(path)

        def log_message(self, format, *args):
            return  # Suppress logging for a cleaner console

    class QuietServer(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    httpd = QuietServer(("127.0.0.1", 0), UmaToolsHandler)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Local server started on port {port}")
    return httpd, thread, port


# --- GUI App ---
class UmalatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UmalatorOCR")
        self.root.geometry("1000x800")

        self.data_manager = DataManager(TOOLS_DIR)
        self.ocr_processor = OCRProcessor(self.data_manager)
        self.url_generator = UmalatorURLGenerator(self.data_manager)

        self.runners: List[Dict[str, str]] = []
        self.monitoring = BooleanVar(value=False)
        self.use_local = BooleanVar(value=True)
        self.custom_url = tk.StringVar(value=UMALATOR_BASE_URL)

        self.runner_strategy = [tk.StringVar(value="Pace Chaser"), tk.StringVar(value="Pace Chaser")]
        self.runner_distance_apt = [tk.StringVar(value="A"), tk.StringVar(value="A")]
        self.runner_surface_apt = [tk.StringVar(value="A"), tk.StringVar(value="A")]
        self.runner_strategy_apt = [tk.StringVar(value="A"), tk.StringVar(value="A")]

        self.ground_var = tk.StringVar(value=GROUND_CONDITIONS[1])
        self.weather_var = tk.StringVar(value=WEATHER_CONDITIONS[0])
        self.season_var = tk.StringVar(value=SEASON_CONDITIONS[0])

        self.httpd = self.server_thread = self.server_port = None
        self.last_csv_mtime = self.focused_canvas = None

        self._create_widgets()
        self.root.after(100, self._init_async)

    def _init_async(self):
        def worker():
            try:
                self.data_manager.load_all_data()
                self.root.after(0, self.load_runners_from_csv)
                self.root.after(0, self._init_dropdowns_default)
            except Exception as e:
                logger.error("Initialization failed: %s", e, exc_info=True)
                self.root.after(0, lambda: messagebox.showerror("Init Error", f"Failed to load data: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self._schedule_csv_watcher()

    def _create_widgets(self):
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Process Screenshots", command=self.start_processing_task).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Refresh List", command=self.load_runners_from_csv).pack(side="left", padx=4)
        ttk.Checkbutton(ctrl, text="Monitor Clipboard", variable=self.monitoring, command=self.toggle_monitoring).pack(
            side="left", padx=8)
        ttk.Checkbutton(ctrl, text="Use Local Server", variable=self.use_local,
                        command=self._toggle_url_entry_state).pack(side="left", padx=8)
        self.url_entry = ttk.Entry(ctrl, textvariable=self.custom_url, width=70)
        self.url_entry.pack(side="left", fill='x', expand=True, padx=2)
        self._toggle_url_entry_state()

        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=10, pady=6)
        left_panel = ttk.LabelFrame(content, text="Select First Runner", padding=6)
        left_panel.pack(side="left", fill="both", expand=True, padx=6)
        right_panel = ttk.LabelFrame(content, text="Select Second Runner", padding=6)
        right_panel.pack(side="right", fill="both", expand=True, padx=6)

        self.left_selection = self._create_selection_area(left_panel)
        self.right_selection = self._create_selection_area(right_panel)
        self._create_runner_controls(left_panel, 0)
        self._create_runner_controls(right_panel, 1)

        bottom = ttk.Frame(self.root, padding=6)
        bottom.pack(fill="x")
        track_ddl_frame = ttk.Frame(bottom)
        track_ddl_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(track_ddl_frame, text="Track:").pack(side="left", padx=(4, 2))
        self.track_var = tk.StringVar()
        self.track_menu = ttk.Combobox(track_ddl_frame, textvariable=self.track_var, state="readonly", width=30)
        self.track_menu.pack(side="left", padx=4)
        self.track_menu.bind("<<ComboboxSelected>>", lambda e: self._on_track_changed())
        ttk.Label(track_ddl_frame, text="Distance:").pack(side="left", padx=(10, 2))
        self.distance_var = tk.StringVar()
        self.distance_menu = ttk.Combobox(track_ddl_frame, textvariable=self.distance_var, state="readonly", width=40)
        self.distance_menu.pack(side="left", padx=4)

        cond_ddl_frame = ttk.Frame(bottom)
        cond_ddl_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(cond_ddl_frame, text="Ground:").pack(side="left", padx=(4, 2))
        ttk.Combobox(cond_ddl_frame, textvariable=self.ground_var, values=GROUND_CONDITIONS, state="readonly",
                     width=12).pack(side="left", padx=4)
        ttk.Label(cond_ddl_frame, text="Weather:").pack(side="left", padx=(10, 2))
        ttk.Combobox(cond_ddl_frame, textvariable=self.weather_var, values=WEATHER_CONDITIONS, state="readonly",
                     width=12).pack(side="left", padx=4)
        ttk.Label(cond_ddl_frame, text="Season:").pack(side="left", padx=(10, 2))
        ttk.Combobox(cond_ddl_frame, textvariable=self.season_var, values=SEASON_CONDITIONS, state="readonly",
                     width=12).pack(side="left", padx=4)

        ttk.Button(bottom, text="Open Umalator Comparison", command=self.open_umalator).pack(side="right", padx=8,
                                                                                             pady=4)

        footer = ttk.Frame(self.root, height=26, style="Footer.TFrame")
        footer.pack(side="bottom", fill="x", ipady=2)
        style = ttk.Style()
        style.configure("Footer.TFrame", background=COLOR_STATUS_BAR_BG)
        style.configure("Footer.TLabel", background=COLOR_STATUS_BAR_BG, foreground=COLOR_STATUS_BAR_FG)
        self.status_label = ttk.Label(footer, text="Ready", anchor="w", style="Footer.TLabel")
        self.status_label.pack(side="left", padx=8, fill="x")

    def _create_selection_area(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return {"frame": content, "selected_idx": None, "selected_widget": None}

    def _create_runner_controls(self, parent, side_index):
        ctl_frame = ttk.Frame(parent)
        ctl_frame.pack(fill="x", pady=(6, 0))
        row1 = ttk.Frame(ctl_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="Strategy:").pack(side="left")
        ttk.Combobox(row1, values=list(STRATEGY_DISPLAY_TO_INTERNAL.keys()),
                     textvariable=self.runner_strategy[side_index], state="readonly", width=16).pack(side="left",
                                                                                                     padx=4)
        ttk.Label(row1, text="Dist Apt:").pack(side="left", padx=(8, 2))
        ttk.Combobox(row1, values=APTITUDES, textvariable=self.runner_distance_apt[side_index], state="readonly",
                     width=6).pack(side="left", padx=4)
        row2 = ttk.Frame(ctl_frame)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="Surf Apt:").pack(side="left")
        ttk.Combobox(row2, values=APTITUDES, textvariable=self.runner_surface_apt[side_index], state="readonly",
                     width=6).pack(side="left", padx=4)
        ttk.Label(row2, text="Strat Apt:").pack(side="left", padx=(8, 2))
        ttk.Combobox(row2, values=APTITUDES, textvariable=self.runner_strategy_apt[side_index], state="readonly",
                     width=6).pack(side="left", padx=4)

    def _schedule_csv_watcher(self):
        try:
            if OUTPUT_CSV_PATH.exists():
                m = OUTPUT_CSV_PATH.stat().st_mtime
                if self.last_csv_mtime is not None and m != self.last_csv_mtime:
                    self.load_runners_from_csv()
                self.last_csv_mtime = m
        finally:
            self.root.after(2000, self._schedule_csv_watcher)

    def load_runners_from_csv(self):
        self.runners = load_csv_rows(OUTPUT_CSV_PATH)
        key_counts = {}
        for r in self.runners:
            key = tuple(r.get(k, "") for k in ["Name", "Epithet", "Speed", "Stamina", "Power", "Guts", "Wit", "Skills"])
            key_counts[key] = key_counts.get(key, 0) + 1

        for side, side_idx in [(self.left_selection, 0), (self.right_selection, 1)]:
            for w in side["frame"].winfo_children(): w.destroy()
            side.update({"selected_idx": None, "selected_widget": None})
            for i, row in enumerate(self.runners):
                widget = self._create_runner_widget(side["frame"], i, row, side_idx, key_counts)
                widget.pack(fill="x", pady=2, padx=2)

        self.update_status(f"Loaded {len(self.runners)} runners.")

    def _create_runner_widget(self, parent, index, row, side_index, key_counts):
        item = tk.Frame(parent, relief="solid", borderwidth=1, bg=COLOR_DEFAULT_BG, padx=5, pady=5)
        name = f"{row.get('Epithet', '')} {row.get('Name', '')}".strip()
        tk.Label(item, text=name, font=("TkDefaultFont", 10, "bold"), bg=COLOR_DEFAULT_BG, anchor="w").pack(fill="x")
        stats_frame = tk.Frame(item, bg=COLOR_DEFAULT_BG)
        stats_frame.pack(fill="x")
        for s in ["Speed", "Stamina", "Power", "Guts", "Wit"]:
            tk.Label(stats_frame, text=f"{s[:3]}: {row.get(s, '')}", font=("TkDefaultFont", 9),
                     bg=COLOR_DEFAULT_BG).pack(side="left", padx=(0, 8))
        if skills := " | ".join(s.strip() for s in row.get("Skills", "").split("|") if s.strip()):
            tk.Label(item, text=skills, bg=COLOR_DEFAULT_BG, wraplength=400, justify="left").pack(fill="x", pady=(4, 0))

        key = tuple(row.get(k, "") for k in ["Name", "Epithet", "Speed", "Stamina", "Power", "Guts", "Wit", "Skills"])
        if key_counts.get(key, 0) > 1:
            for widget in (item, *item.winfo_children()): widget.configure(bg=COLOR_DUP_BG)

        handler = lambda e, i=index, w=item, s_idx=side_index: self._select_runner(i, w, s_idx)

        def bind_recursive(w):
            w.bind("<Button-1>", handler)
            for ch in w.winfo_children(): bind_recursive(ch)

        bind_recursive(item)
        return item

    def _select_runner(self, index, widget, side_index):
        side = self.left_selection if side_index == 0 else self.right_selection
        if side["selected_widget"]: self._set_recursive_bg(side["selected_widget"], COLOR_DEFAULT_BG, COLOR_DUP_BG)
        self._set_recursive_bg(widget, COLOR_SELECTED_BG, COLOR_SELECTED_BG)
        side["selected_idx"], side["selected_widget"] = index, widget

        runner = self.runners[index]
        self.runner_surface_apt[side_index].set(runner.get("surfaceAptitude", "A"))
        self.runner_distance_apt[side_index].set(runner.get("distanceAptitude", "A"))
        self.runner_strategy_apt[side_index].set(runner.get("strategyAptitude", "A"))
        display_strategy = STRATEGY_INTERNAL_TO_DISPLAY.get(runner.get("strategy", "Senkou"), "Pace Chaser")
        self.runner_strategy[side_index].set(display_strategy)
        self.update_status(f"Selected {runner.get('Name', 'Unknown')} for {'left' if side_index == 0 else 'right'}.")

    def _set_recursive_bg(self, widget, color, dup_color):
        is_dup = widget.cget("bg") == COLOR_DUP_BG
        new_color = dup_color if is_dup and color == COLOR_DEFAULT_BG else color
        try:
            widget.configure(bg=new_color)
            for child in widget.winfo_children():
                self._set_recursive_bg(child, color, dup_color)
        except tk.TclError:
            pass

    def update_status(self, message):
        if self.root.winfo_exists(): self.root.after(0, lambda: self.status_label.config(text=message))

    def start_processing_task(self):
        self.update_status("Processing screenshots...")
        threading.Thread(target=self._process_screenshots_worker, daemon=True).start()

    def _process_screenshots_worker(self):
        try:
            count = process_screenshot_folder(self.ocr_processor)
            self.update_status(f"Processing complete. Added {count} new runners.")
            self.root.after(0, self.load_runners_from_csv)
        except Exception as e:
            logger.error("Processing failed: %s", e, exc_info=True)
            self.update_status(f"Error: {e}")

    def toggle_monitoring(self):
        if self.monitoring.get():
            self.stop_monitoring = False
            threading.Thread(target=self._monitor_clip_worker, daemon=True).start()
            self.update_status("Clipboard monitoring enabled.")
        else:
            self.stop_monitoring = True
            self.update_status("Clipboard monitoring disabled.")

    def _monitor_clip_worker(self):
        while not getattr(self, "stop_monitoring", False):
            if monitor_clipboard(self.ocr_processor, self.update_status):
                self.root.after(0, self.load_runners_from_csv)
            time.sleep(1)

    def _toggle_url_entry_state(self):
        self.url_entry.config(state="disabled" if self.use_local.get() else "normal")

    def _init_dropdowns_default(self):
        tracks = sorted([(str(k), v[1] if isinstance(v, list) and len(v) > 1 else v[0]) for k, v in
                         self.data_manager.track_names.items() if v], key=lambda x: x[1])
        self.track_items = list(dict.fromkeys(tracks))
        self.track_menu['values'] = [d for k, d in self.track_items]

        default_track_key = next((str(e.get("raceTrackId")) for c, e in self.data_manager.course_data.items() if
                                  str(c) == str(DEFAULT_COURSE_ID)), None)
        default_display = next((d for k, d in self.track_items if k == default_track_key),
                               self.track_items[0][1] if self.track_items else "")
        self.track_var.set(default_display)
        self._on_track_changed()

    def _on_track_changed(self):
        sel_key = next((k for k, d in self.track_items if d == self.track_var.get()), None)
        if not sel_key: return

        entries = self.data_manager.track_index.get(sel_key, [])
        self.distance_map = {}
        for e in entries:
            dist, surf = e.get("distance", 0), e.get("surface", 0)
            cid_e = next((cid for cid, val in self.data_manager.course_data.items() if val is e), None)
            if cid_e: self.distance_map[f"{dist}m - {'Turf' if surf == 1 else 'Dirt'}"] = cid_e

        dist_list = sorted(self.distance_map.keys(), key=lambda s: int(re.match(r"(\d+)", s).group(1)))
        self.distance_menu['values'] = dist_list

        default_dist = next((d for d, c in self.distance_map.items() if c == str(DEFAULT_COURSE_ID)),
                            dist_list[0] if dist_list else "")
        self.distance_var.set(default_dist)

    def open_umalator(self):
        if self.left_selection["selected_idx"] is None or self.right_selection["selected_idx"] is None:
            return messagebox.showwarning("Selection Required", "Please select two runners.")

        cid_str = self.distance_map.get(self.distance_var.get())
        if not cid_str: return messagebox.showerror("Course Error", "Please select a valid track and distance.")

        def copy_and_inject(row_idx, side):
            r = self.runners[row_idx].copy()
            r["strategy"] = STRATEGY_DISPLAY_TO_INTERNAL.get(self.runner_strategy[side].get(), "Senkou")
            r["distanceAptitude"] = self.runner_distance_apt[side].get()
            r["surfaceAptitude"] = self.runner_surface_apt[side].get()
            r["strategyAptitude"] = self.runner_strategy_apt[side].get()
            return r

        r1 = copy_and_inject(self.left_selection["selected_idx"], 0)
        r2 = copy_and_inject(self.right_selection["selected_idx"], 1)

        try:
            share = self.url_generator.create_url_from_rows(
                [r1, r2], course_id=int(cid_str),
                ground=GROUND_MAP.get(self.ground_var.get(), 1),
                weather=WEATHER_MAP.get(self.weather_var.get(), 1),
                season=SEASON_MAP.get(self.season_var.get(), 1))
        except Exception as e:
            logger.error("Failed to create URL: %s", e, exc_info=True)
            return messagebox.showerror("Error", f"Failed to create share URL: {e}")

        if self.use_local.get():
            if self.httpd is None:
                try:
                    self.httpd, self.server_thread, self.server_port = start_local_server()
                except Exception as e:
                    return messagebox.showerror("Server Error", f"Failed to start local server: {e}")
            base = f"http://127.0.0.1:{self.server_port}/"
        else:
            base = self.custom_url.get().rstrip("/") + "/"

        webbrowser.open_new_tab(f"{base}index.html#{share}")
        self.update_status("Opened Umalator comparison in browser.")
        return None

    def on_closing(self):
        self.stop_monitoring = True
        if self.httpd: self.httpd.shutdown()
        self.root.destroy()


# --- process screenshot folder helper ---
def process_screenshot_folder(ocr_processor: OCRProcessor) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    count = 0

    for p in images:
        try:
            logger.debug("Processing file: %s", p)
            row = ocr_processor.extract_data_from_image(str(p), use_text_matching=True)
            if row and row.get("Name") and row.get("Name") != "Unknown":
                append_to_csv(row, OUTPUT_CSV_PATH)
                PROCESSED_DIR.mkdir(exist_ok=True)
                p.rename(PROCESSED_DIR / p.name)
                count += 1
        except IOError:
            logger.error("Failed to read image file: %s", p)
        except Exception as e:
            logger.error("Failed to process %s: %s", p.name, str(e))
    return count


# --- main ---
def main():
    root = tk.Tk()
    app = UmalatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
