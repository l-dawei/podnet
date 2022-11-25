import mindspore.dataset as ds
import mindspore as ms

dataset_path = 'D:\\file\Dataset\\flower\\flower_photos'
dataset = ds.ImageFolderDataset(dataset_dir=dataset_path)
print(dataset)

trans = [
    ds.vision.c_transforms.Decode(),
    ds.vision.c_transforms.Resize(256),
    ds.vision.c_transforms.CenterCrop(256)
]
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
trans_norm = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]
type_cast_op = ds.transforms.c_transforms.TypeCast(ms.int32)
dataset = dataset.map(operations=trans, input_columns="image")
dataset = dataset.map(operations=trans_norm, input_columns="image")
data_set = dataset.map(operations=type_cast_op, input_columns="label")
for i, data in enumerate(dataset.create_dict_iterator()):
    # print(data['label'])
    print(data['image'].shape)
#     # print(data.keys())  # dict_keys(['image', 'label'])
#
#     if i >= 10:
#         break

# train_ds.map(operations=ds.vision.c_transforms.Resize(224, 224))


# from PIL import Image
# x =
# img = Image.open(x).convert("RGB")