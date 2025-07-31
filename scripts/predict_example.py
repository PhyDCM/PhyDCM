from phydcm import PyHDCMPredictor 

if __name__ == "__main__":
    predictor = PyHDCMPredictor()
    image_path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_16.bmp"
    image_1path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_34.bmp"
    image_2path = r"C:\Users\lenovo\Desktop\phydcm\data/pet/val/follicular_lymphoma/image_45.bmp"
    scan1_type = "pet"
    scan2_type = "pet"

    r3esult = predictor.predict(image_path, scan1_type)
    r1esult = predictor.predict(image_1path, scan2_type)
    r2esult = predictor.predict(image_2path, scan2_type)
    
    print(f"Prediction for {scan1_type.upper()}: {r3esult}")
    print(f"Prediction for {scan2_type.upper()}: {r1esult}")
    print(f"Prediction for {scan2_type.upper()}: {r2esult}")

