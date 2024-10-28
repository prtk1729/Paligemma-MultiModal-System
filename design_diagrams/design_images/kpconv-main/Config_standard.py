import yaml
from utils.config import Config
class Config_standard(Config):
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)


        # Dataset parameters
        self.dataset = config_dict['dataset']
        self.num_classes = config_dict['num_classes']
        self.dataset_task = config_dict['dataset_task']
        self.input_threads = config_dict['input_threads']

        # Architecture definition
        self.architecture = config_dict['architecture']

        # KPConv parameters
        self.num_kernel_points = config_dict['num_kernel_points']
        self.in_radius = config_dict['in_radius']
        self.first_subsampling_dl = config_dict['first_subsampling_dl']
        self.conv_radius = config_dict['conv_radius']
        self.deform_radius = config_dict['deform_radius']
        self.KP_extent = config_dict['KP_extent']
        self.KP_influence = config_dict['KP_influence']
        self.aggregation_mode = config_dict['aggregation_mode']
        self.first_features_dim = config_dict['first_features_dim']
        self.in_features_dim = config_dict['in_features_dim']
        self.modulated = config_dict['modulated']
        self.use_batch_norm = config_dict['use_batch_norm']
        self.batch_norm_momentum = config_dict['batch_norm_momentum']
        self.deform_fitting_mode = config_dict['deform_fitting_mode']
        self.deform_fitting_power = config_dict['deform_fitting_power']
        self.deform_lr_factor = config_dict['deform_lr_factor']
        self.repulse_extent = config_dict['repulse_extent']

        # Training parameters
        self.num_votes = config_dict['num_votes']
        self.max_epoch = config_dict['max_epoch']
        self.learning_rate = config_dict['learning_rate']
        self.momentum = config_dict['momentum']
        self.lr_decays = config_dict['lr_decays']
        self.grad_clip_norm = config_dict['grad_clip_norm']
        self.batch_num = config_dict['batch_num']
        self.epoch_steps = config_dict['epoch_steps']
        self.validation_size = config_dict['validation_size']
        self.checkpoint_gap = config_dict['checkpoint_gap']
        self.augment_scale_anisotropic = config_dict['augment_scale_anisotropic']
        self.augment_symmetries = config_dict['augment_symmetries']
        self.augment_rotation = config_dict['augment_rotation']
        self.augment_scale_min = config_dict['augment_scale_min']
        self.augment_scale_max = config_dict['augment_scale_max']
        self.augment_noise = config_dict['augment_noise']
        self.augment_color = config_dict['augment_color']
        self.segloss_balance = config_dict['segloss_balance']
        self.saving = config_dict['saving']
        self.saving_path = config_dict['saving_path']
        self.dataset_path_train = config_dict['dataset_path_train']
        self.dataset_path_test = config_dict['dataset_path_test']
        self.dataset_path_validation = config_dict['dataset_path_validation']
        self.path_output = config_dict['path_output']
        self.ignored_labels = config_dict['ignored_labels']
        self.label_continue = config_dict['label_continue']
        self.label_to_names = config_dict['label_to_names']
        super().__init__()
