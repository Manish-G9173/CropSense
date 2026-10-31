"""Safe decoding of an uploaded leaf image or first PDF page."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PDF_RENDER_SCALE = 1.5


class PlantUploadError(ValueError):
    """Raised when an uploaded photo/PDF cannot be used for diagnosis."""


def decode_plant_upload(filename: str, data: bytes) -> tuple[Image.Image, str]:
    """Return a normalized image and a note describing PDF page handling."""
    if not data:
        raise PlantUploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise PlantUploadError("The file is over 20 MB. Please upload a smaller image or PDF.")

    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise PlantUploadError(
                "PDF support is missing. Install the disease extras from requirements-disease.txt."
            ) from exc
        try:
            document = pdfium.PdfDocument(data)
            if len(document) < 1:
                raise PlantUploadError("The PDF has no pages.")
            page = document[0]
            bitmap = page.render(scale=PDF_RENDER_SCALE)
            image = bitmap.to_pil().convert("RGB")
            page.close()
            document.close()
            return ImageOps.exif_transpose(image), "Only page 1 of the PDF was used."
        except PlantUploadError:
            raise
        except Exception as exc:
            raise PlantUploadError("Could not read this PDF. Try saving it again or uploading a JPG.") from exc

    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise PlantUploadError("Use a JPG, JPEG, PNG, WEBP, or PDF file.")
    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        with Image.open(BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB"), ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PlantUploadError("This image could not be opened. Try exporting it as JPG or PNG.") from exc
