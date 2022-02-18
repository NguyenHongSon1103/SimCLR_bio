config = dict(
    batch_size = 32,
    steps_per_epoch = None,
    n_classes = 10,
    data_A = '',
    data_B = '',
    optimizer_type = 'sgd', # one of ['sgd', 'adam']
    use_lr_schedule = True,
    base_lr = 1e-3,
    end_lr = 1e-6,
    gpu = '1',
    resume = False,
    pretrained_weights = '',

)