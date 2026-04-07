from app.config import get_settings
from app.ml.training import train_and_save_model


if __name__ == "__main__":
    settings = get_settings()
    result = train_and_save_model(
        model_dir=settings.model_dir,
        data_dir=settings.data_dir,
        dataset_url=settings.model_data_url,
        target_column=settings.model_target_column,
    )
    print(f"Saved model to {result.artifact_path}")
