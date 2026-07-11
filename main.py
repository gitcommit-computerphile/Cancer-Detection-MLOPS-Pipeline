from cancer.pipeline.training_pipeline import TrainingPipeline
from cancer.logging.logger import logger

if __name__ == "__main__":
    logger.info("Starting cancer detection training pipeline")
    pipeline = TrainingPipeline()
    artifact = pipeline.run_pipeline()
    logger.info(f"Pipeline complete: {artifact}")
    print(f"\nTrain F1: {artifact.train_metric_artifact.f1_score:.4f}")
    print(f"Test  F1: {artifact.test_metric_artifact.f1_score:.4f}")