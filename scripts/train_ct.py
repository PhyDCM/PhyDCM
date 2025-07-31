from phydcm.train import train_model

if __name__ == "__main__":
    import sys
    modality = 'ct'  # أو خذها من sys.argv لو تحب

    train_model(modality)
