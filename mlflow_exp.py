import mlflow

mlflow.set_experiment("rag_experiment")

with mlflow.start_run():
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("chunk_overlap", 100)
    mlflow.log_metric("accuracy", 0.85)  # Example

    print("Experiment logged")