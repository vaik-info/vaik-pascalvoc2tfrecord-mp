# vaik-pascalvoc2tfrecord-mp

Parallel convert from Pascal VOC format to tfrecord format for [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Example

![vaik-pascalvoc2tfrecord-mp](https://user-images.githubusercontent.com/116471878/198183072-c26bac61-1652-40c9-a2f6-9100bb12fb2c.png)


## Usage

```shell
pip install -r requirements.txt
python main.py --input_image_dir_path ~/.vaik-mnist-detection-dataset/train \
                --input_classes_path ~/.vaik-mnist-detection-dataset/classes.txt \
                --output_dir_path ~/.vaik-pascalvoc2tfrecord-mp/vaik-mnist-detection-dataset/train \
                --records_prefix_index 00 \
                --cpu_count 32
```

## Output

![vaik-pascalvoc2tfrecord-mp-out](https://user-images.githubusercontent.com/116471878/198181347-5ae5a8d5-f336-4643-b93f-2c1edb63f54b.png)