from phydcm import PyHDCMPredictor 

if __name__ == "__main__":
    predictor = PyHDCMPredictor()
    image_path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_16.bmp"
    image_1path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_34.bmp"
    image_2path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_45.bmp"
    image_3path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_7.bmp"
    scan_type = "pet"
    
    result = predictor.predict(image_path, scan_type)
    r1esult = predictor.predict(image_1path, scan_type)
    r2esult = predictor.predict(image_2path, scan_type)
    r3esult = predictor.predict(image_3path, scan_type)
    print(f"Prediction for {scan_type.upper()}: {result}")
    print(f"Prediction for {scan_type.upper()}: {r1esult}")
    print(f"Prediction for {scan_type.upper()}: {r2esult}")
    print(f"Prediction for {scan_type.upper()}: {r3esult}")
