"""
classifier/clip_classifier.py
------------------------------
Loads the CLIP model ONCE at startup and classifies PIL images.

Returns per image:
  - prediction      : class label or "Review Needed"
  - confidence      : float 0-1
  - xray_score      : raw softmax score
  - prescription_score
  - other_score
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.settings import CLIP_MODEL_NAME, CONFIDENCE_THRESHOLD
from utils.logger import get_logger

log = get_logger("classifier")

# ── Class Labels ──────────────────────────────────────────────────────────────

CLASSES = ["X-ray / Sonography", "Prescription / Document", "Other"]

# ── Prompts 
# Multiple prompts per class → averaged into one embedding.
# More prompts = more robust representation of each class.

XRAY_PROMPTS = [
    "an X-ray radiograph of bones or chest with dark background and no face",
    "a black and white X-ray image showing skeletal structure or joints",
    "an ultrasound sonography image with speckled grayscale noise pattern",
    "a grayscale MRI or CT scan slice showing internal organs or brain cross section",
    "a chest X-ray showing ribs lungs or spine on a lightbox",
    "a medical radiology scan with no text overlay and no human face",
    "a sonography image showing fetus kidney liver or gallbladder",
    "an X-ray of hand foot knee or spine showing bone structure",
    "a darkfield medical imaging scan with bright white bone structures",
    "an echocardiogram or doppler ultrasound image with colored flow patterns",
]

PRESCRIPTION_PROMPTS = [
    "a printed medical prescription paper with medicine names and dosage",
    "a handwritten doctor prescription slip with patient details and drug names",
    "a typed medical lab test report on white paper with test values and ranges",
    "a medical bill or discharge summary document with text and tables",
    "a pharmacy prescription with medicine quantities and instructions",
    "a pathology report or blood test result printed on white paper",
    "a medical certificate or fitness certificate with doctor signature and stamp",
    "a written prescription with Rx symbol and multiple medicine names listed",
    "a clinical document with rows of text numbers and doctor handwriting",
    "a diagnostic report or radiology report printed as a text document",
]

OTHER_PROMPTS = [
    "a close-up color photograph of a human face with visible eyes nose mouth and skin",
    "a portrait of a person showing face with natural skin tone and hair",
    "a selfie photograph of a persons face taken with a phone camera",
    "a color photo of a persons head with visible facial features and skin texture",
    "a photograph of a smiling person showing teeth eyes and face clearly",
    "a color photograph of a human hand showing fingers and skin texture",
    "a photograph of a human leg foot or arm with visible skin",
    "a color photo of a body part like elbow knee wrist or shoulder with skin",
    "a wound or skin condition photograph showing skin surface in color",
    "a photograph of a rash bruise or skin lesion on colored skin",
    "a signatue photograph",
    "a company or app logo icon on a plain or colored background",
    "a digital splash screen or launch screen showing a brand logo",
    "a graphic logo image with a symbol and brand name, not a document",
    "an application icon or watermark image with minimal text",
]

ALL_PROMPTS = [XRAY_PROMPTS, PRESCRIPTION_PROMPTS, OTHER_PROMPTS]


class CLIPClassifier:
    """
    Loads CLIP once. Call .classify(pil_image) for each image.
    Thread-safe for read-only inference.
    """

    def __init__(self):
        log.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
        self.model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info("CLIP model loaded on device: %s", self.device)

        self._text_embeds = self._build_text_embeddings()
        log.info("Text embeddings built for %d classes.", len(CLASSES))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _avg_text_embedding(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        with torch.no_grad():
            embeds = self.model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        avg    = embeds.mean(dim=0, keepdim=True)
        return avg / avg.norm(dim=-1, keepdim=True)

    def _build_text_embeddings(self) -> torch.Tensor:
        return torch.cat(
            [self._avg_text_embedding(p) for p in ALL_PROMPTS], dim=0
        )  # shape: [num_classes, 512]

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, image: Image.Image) -> dict:
        """
        Classify a single PIL image.

        Returns:
            {
                "prediction":          str,   # class label or "Review Needed"
                "confidence":          float,
                "xray_score":          float,
                "prescription_score":  float,
                "other_score":         float,
            }
        """
        try:
            inputs = self.processor(
                images=image, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                img_embed = self.model.get_image_features(**inputs)

            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

            logits = (img_embed @ self._text_embeds.T) * self.model.logit_scale.exp()
            probs  = logits.softmax(dim=1)[0]  # shape: [num_classes]

            pred_idx   = probs.argmax().item()
            confidence = probs[pred_idx].item()

            label = CLASSES[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else "Review Needed"

            return {
                "prediction":         label,
                "confidence":         round(confidence, 4),
                "xray_score":         round(probs[0].item(), 4),
                "prescription_score": round(probs[1].item(), 4),
                "other_score":        round(probs[2].item(), 4),
            }

        except Exception as e:
            log.error("CLIP inference failed: %s", e)
            return {
                "prediction":         "ERROR",
                "confidence":         0.0,
                "xray_score":         0.0,
                "prescription_score": 0.0,
                "other_score":        0.0,
            }
