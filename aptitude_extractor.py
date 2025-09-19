import logging
import json
from typing import List, Dict, Tuple

logger = logging.getLogger("UmalatorOCR")

# Rank ordering for aptitudes
GRADE_ORDER = {"S": 7, "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}

STYLE_MAP = {
    "Front": "Nige",
    "Pace": "Senkou",
    "Late": "Sashi",
    "End": "Oikomi",
}


class AptitudeExtractor:
    DEBUG_FILE = "aptitude_debug.json"

    @staticmethod
    def extract_from_ocr_results(
        res: List[Tuple[List[Tuple[int, int]], str, float]]
    ) -> Dict[str, str]:
        """
        Given OCR results [(box, text, conf), ...], extract the best surface, distance, and style aptitudes.
        Also writes a debug JSON file with raw row/grade findings.
        """
        texts = [t.strip() for _, t, _ in res if t.strip()]
        logger.debug("Extracting aptitudes from OCR texts: %s", texts)

        debug_info = {"surface": [], "distance": [], "style": []}

        # --- Surface (row 1) ---
        surface_grades = [t for t in texts if t in GRADE_ORDER]
        surface = AptitudeExtractor._pick_best(surface_grades)
        debug_info["surface"] = surface_grades

        # --- Distance (row 2) ---
        distance_grades = [t for t in texts if t in GRADE_ORDER]
        distance = AptitudeExtractor._pick_best(distance_grades)
        debug_info["distance"] = distance_grades

        # --- Style (row 3) ---
        style = None
        style_grade = "A"  # fallback
        style_candidates = []
        for i, t in enumerate(texts):
            if t in STYLE_MAP:
                if i + 1 < len(texts) and texts[i + 1] in GRADE_ORDER:
                    grade = texts[i + 1]
                    style_candidates.append((t, grade))
                    if GRADE_ORDER[grade] > GRADE_ORDER[style_grade]:
                        style_grade = grade
                        style = STYLE_MAP[t]

        if not style:
            style = "Nige"  # default if none found

        debug_info["style"] = style_candidates

        # --- Write debug log ---
        try:
            with open(AptitudeExtractor.DEBUG_FILE, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            logger.debug("Wrote aptitude debug info to %s", AptitudeExtractor.DEBUG_FILE)
        except Exception as e:
            logger.error("Failed to write aptitude debug info: %s", e)

        # --- Console-friendly summary log ---
        logger.info(
            f"SurfaceApt: {surface}, DistanceApt: {distance}, "
            f"StyleApt: {style_grade}, Style: {style}"
        )

        return {
            "surfaceAptitude": surface,
            "distanceAptitude": distance,
            "strategyAptitude": style_grade,
            "strategy": style,
        }

    @staticmethod
    def _pick_best(grades: List[str]) -> str:
        if not grades:
            return "A"
        return max(grades, key=lambda g: GRADE_ORDER.get(g, -1))
