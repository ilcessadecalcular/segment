class hparams:

    GPUS = (0,)



    shuffle = True
    workers=0
    train_or_test = 'train'
    output_dir = 'logs/hrnetrnn84'
    aug = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 150
    epochs_per_checkpoint = 10
    batch_size = 1
    ckpt = None
    init_lr = 0.0002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_size = 48


    crop_or_pad_size = 150,256,256 # if 2D: 256,256,1
    patch_size = 110,128,32 # if 2D: 128,128,1

    divide = 16

    
    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'

    source_train_dir = 'train/image'
    label_train_dir = 'train/label'

    source_eval_dir = 'eval/image'
    label_eval_dir = 'eval/label'

    source_test_dir = 'test/image'
    label_test_dir = 'test/label'
    model_test_dir = 'logs/hrnetrnn3+81'
    latest_checkpoint_file = 'checkpoint_latest.pt'



    output_dir_test = 'results/83'



