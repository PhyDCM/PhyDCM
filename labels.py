import json
labels = {
    "0": "glioma",
    "1": "meningioma",
    "2": "notumor",
    "3": "pituitary"
}

with open('outputs/mri_labels.json', 'w') as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)
print("✅ تم إنشاء ملف labels json بنجاح")
