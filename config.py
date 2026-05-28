import os


class Config:
    def __init__(self, config_path=None):
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.mintpain_root = os.path.abspath(os.path.join(project_root, "..", "dataset", "mintpain"))
        self.preprocessed_dir = os.path.join(self.mintpain_root, "rgb_preprocessed")
        self.loso_splits_path = os.path.join(self.mintpain_root, "loso_splits.pkl")
        self.samples_pkl_path = os.path.join(self.mintpain_root, "mintpain_edlm_samples.pkl")

        self.features_dir = os.path.join(self.mintpain_root, "edlm_features")
        self.features_4d_dir = os.path.join(self.features_dir, "4d")
        self.features_3d_dir = os.path.join(self.features_dir, "3d")
        self.weights_dir = os.path.join(self.mintpain_root, "weights")
        self.output_dir = os.path.join(self.mintpain_root, "results")

        self.vggface_weights_path = os.path.join(self.weights_dir, "vgg_face_dag.pth")
        self.feature_backbone = "vgg16"

        self.num_classes = 5
        self.sequence_length = 5
        self.bottleneck_dim = 4
        self.pca_dim = 3
        self.undersample = True

        self.feature_extractor_lr = 0.001
        self.feature_extractor_backbone_lr = 0.0
        self.feature_extractor_batch_size = 192
        self.feature_extractor_epochs = 50

        self.ensemble_lr = 0.001
        self.ensemble_batch_size = 192
        self.ensemble_epochs = 5

        self.num_folds = 0
        self.device = "cuda"

        for directory in [
            self.features_dir, self.features_4d_dir, self.features_3d_dir,
            self.weights_dir, self.output_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
