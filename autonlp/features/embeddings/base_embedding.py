from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Base_Embedding:
    """ Parent class of embedding methods """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        """
        Args:
            flags_parameters : Instance of Flags class object
            column_text (int) : column number with texts
            dimension_embedding (str) : 'word_embedding' or 'doc_embedding'

        From flags_parameters:
        objective (str) : 'binary' or 'multi-class' or 'regression'
        average_scoring (str) : 'micro', 'macro' or 'weighted'
        seed (int)
        apply_mlflow (Boolean)
        experiment_name (str)
        """
        self.flags_parameters = flags_parameters
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.seed = flags_parameters.seed
        self.column_text = column_text
        self.name_model = None
        self.dimension_embedding = dimension_embedding

        if self.apply_mlflow:
            import mlflow
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

    def hyper_params(self, size_params='small'):
        pass

    def preprocessing_fit_transform(self, **kwargs):
        pass

    def preprocessing_transform(self, **kwargs):
        pass
