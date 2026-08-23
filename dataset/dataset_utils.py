shapes_in = {
    'cifar10': (3, 32, 32),
    'cinic10': (3, 32, 32),
    'cifar10c': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'digit': (3, 28, 28),
    'pacs_aug': (3, 224, 224),
    'imagenet': (3, 224, 224),
    'office31_aug': (3, 224, 224),
    'office_home_aug': (3, 224, 224),
    'tiny_imagenet': (3, 224, 224),
    'office_caltech10_aug': (3, 224, 224),
    'domainnet_aug': (3, 224, 224),  # 新增DomainNet
    'food101': (3, 224, 224),
    'stanfordcars': (3, 224, 224),
}

shapes_out = {
    'cifar10': 10,
    'cinic10': 10,
    'cifar10c': 10,
    'cifar100': 100,
    'digit': 10,
    'pacs_aug': 7,
    'imagenet': 1000,
    'office31_aug': 31,
    'office_home_aug': 65,
    'tiny_imagenet': 200,
    'office_caltech10_aug': 10,
    'domainnet_aug': 345,  # 新增DomainNet，345个类别
    'food101': 101,
    'stanfordcars': 196,
}