

# configs for HRNet48
class HRNet48:

    FINAL_CONV_KERNEL = 1
    class STAGE1:
        NUM_MODULES = 1
        NUM_BRANCHES = 1
        NUM_BLOCKS = [4]
        NUM_CHANNELS = [64]
        BLOCK = 'BOTTLENECK'
        FUSE_METHOD = 'SUM'

    class STAGE2:
        NUM_MODULES = 1
        NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [48, 96]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE3:
        NUM_MODULES = 4
        NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [48, 96, 192]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE4:
        NUM_MODULES = 3
        NUM_BRANCHES = 4
        NUM_BLOCKS = [4, 4, 4, 4]
        NUM_CHANNELS = [48, 96, 192, 384]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class DATASET:
        NUM_CLASSES = 48
        
        
        
# configs for HRNet32
class HRNet32:
    PRETRAINED = ''
    FINAL_CONV_KERNEL = 1
    class STAGE1:
        NUM_MODULES = 1
        NUM_BRANCHES = 1
        NUM_BLOCKS = [4]
        NUM_CHANNELS = [64]
        BLOCK = 'BOTTLENECK'
        FUSE_METHOD = 'SUM'

    class STAGE2:
        NUM_MODULES = 1
        NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [32, 64]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE3:
        NUM_MODULES = 4
        NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [32, 64, 128]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE4:
        NUM_MODULES = 3
        NUM_BRANCHES = 4
        NUM_BLOCKS = [4, 4, 4, 4]
        NUM_CHANNELS = [32, 64, 128, 256]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class DATASET:
        NUM_CLASSES = 32
        
        
        
        
# configs for HRNet18
class HRNet18:
    PRETRAINED = ''
    FINAL_CONV_KERNEL = 1
    class STAGE1:
        NUM_MODULES = 1
        NUM_BRANCHES = 1
        NUM_BLOCKS = [4]
        NUM_CHANNELS = [64]
        BLOCK = 'BOTTLENECK'
        FUSE_METHOD = 'SUM'

    class STAGE2:
        NUM_MODULES = 1
        NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [18, 36]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE3:
        NUM_MODULES = 1
        NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [18, 36, 72]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class STAGE4:
        NUM_MODULES = 1
        NUM_BRANCHES = 4
        NUM_BLOCKS = [4, 4, 4, 4]
        NUM_CHANNELS = [18, 36, 72, 144]
        BLOCK = 'BASIC'
        FUSE_METHOD = 'SUM'

    class DATASET:
        NUM_CLASSES = 18
