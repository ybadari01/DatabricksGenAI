# demo_setup/experiment_manager.py

import mlflow
from mlflow import MlflowClient


class ExperimentManager:
    """
    Manages MLflow experiments.
    
    Responsible for:
    - Creating and deleting experiments
    - Managing experiment lifecycle
    """
    
    def __init__(self, username: str):
        """
        Initialize experiment manager.
        
        Parameters
        ----------
        username : str
            Databricks username for experiment path
        """
        self.username = username
        mlflow.set_tracking_uri("databricks")
    
    def create_experiment(self, experiment_name: str) -> str:
        """
        Create or recreate an experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
            
        Returns
        -------
        str
            Full experiment path
        """
        print(f"Setting up experiment: {experiment_name}")
        experiment_path = f"/Workspace/Users/{self.username}/{experiment_name}"

        exp = mlflow.get_experiment_by_name(experiment_path)
        if exp is not None:
            print(f"Experiment exists. Deleting and recreating: {experiment_name}")
            mfc = MlflowClient()
            mfc.delete_experiment(exp.experiment_id)
            print(f"✅ Deleted experiment {exp.experiment_id}")

        mlflow.create_experiment(experiment_path)
        print(f"✅ Created experiment: {experiment_path}")
        
        return experiment_path
    
    def set_experiment(self, experiment_path: str) -> None:
        """Set the active MLflow experiment."""
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_path)