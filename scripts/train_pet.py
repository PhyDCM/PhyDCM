from phydcm.train import train_model

if __name__ == "__main__":
    import sys
    modality = 'pet'  # أو خذها من sys.argv لو تحب

    train_model(modality)
