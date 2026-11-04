from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


# Data Ingestion
ingestion = DataIngestion()
ingestion.initiate_data_ingestion()

# Data Validation
validation = DataValidation()
validation.initiate_data_validation()

# Data Transformation
transformation = DataTransformation()
transformation.initiate_data_transformation()

# Model Training
trainer = ModelTrainer()

(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
) = trainer.initiate_model_trainer()

# Model Evaluation
evaluation = ModelEvaluation()

comparison_df = evaluation.initiate_model_evaluation(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
)

print("\nPipeline Completed Successfully!")
print(comparison_df)