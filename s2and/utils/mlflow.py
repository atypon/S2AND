import mlflow


def get_or_create_experiment(name: str) -> str:
    """
    Creates mlflow experiment with specified name of
    retrieves it if it is already created or deleted
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=name)
    if experiment is None:
        experiment_id = client.create_experiment(name=name)
    else:
        if dict(experiment)['lifecycle_stage'] == 'deleted':
            client.restore_experiment(dict(experiment)['experiment_id'])
        experiment_id = dict(experiment)['experiment_id']
    return experiment_id
