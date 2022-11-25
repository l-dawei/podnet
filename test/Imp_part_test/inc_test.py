from inclearn.lib.data.incdataset import IncrementalDataset

inc_dataset = IncrementalDataset(
    dataset_name='cifar100',
    random_order=False,
    shuffle=True,
    batch_size=128,
    workers=1,
    validation_split=0.0,
    onehot=False,
    increment=1,
    initial_increment=50,
    sampler=None,
    sampler_config={},
    data_path='D://file//Dataset//cifar-100-binary',
    class_order=
    [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
     4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
     53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2,
     95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39],
    seed=0,
    dataset_transforms={'color_jitter': True},
    all_test_classes=False,
    metadata_path=None
)

memory, memory_val = None, None
task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)

# print(task_info)
# {'min_class': 0, 'max_class': 50, 'total_n_classes': 100, 'increment': 50, 'task': 0,
# 'max_task': 51, 'n_train_data': 25000, 'n_test_data': 25000}
# print(val_loader) # None
# print(train_loader)
# print(test_loader)
# print(train_loader)
# step_size = train_loader.get_dataset_size()
# print(step_size)  # step_size==196, because n_train_data == 25000 == 195 * 128 + 40(last batch only has 40 pic)
dict_datasets = next(train_loader.create_dict_iterator())
# print(type(dict_datasets))  # dict
# print(dict_datasets.keys())  # dict_keys(['image', 'label', 'memory_flag'])
print(dict_datasets['image'].shape)  # tensor (128, 3, 32, 32)
# print(dict_datasets['label'].shape)  # tensor (128, )
# print(dict_datasets['memory_flag'].shape)  # tensor (128, )

# for i, data in enumerate(train_loader.create_dict_iterator()):  # i is from 0 to max_num
#     if i == 195:
#         print(data['image'].shape)
# print(i)
