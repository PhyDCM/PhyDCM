"""
Example: running inference with PhyDCM.

Update `model_dir` to point to the directory that contains:
- mri_best_model.keras / ct_best_model.keras / pet_best_model.keras (or equivalent supported names)
- mri_labels.json / ct_labels.json / pet_labels.json
"""
from __future__ import annotations

from phydcm import PyHDCMPredictor


def main() -> None:
    predictor = PyHDCMPredictor(
        model_dir=None,   # uses packaged 'phydcm/models' by default
        img_size=(224, 224),
        auto_load=True,
    )

    image_path = r"E:\PhyDCM-main\data\mri\val\glioma\Tr-gl_0021.jpg"
    scan_type = "mri"  # options: "mri", "ct", "pet"

    result = predictor.predict(image_path, scan_type, return_probabilities=True)
    print(result)


if __name__ == "__main__":
    main()
