import os
import tensorflow as tf
import random

from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def pascal_voc2tf_example(image_path, xml_path, classes, random_quality=(90, 100)):
    tf_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    encoded_image_data = tf.io.encode_jpeg(tf_image,
                                           quality=int(random.uniform(random_quality[0], random_quality[1]))).numpy()
    xml_dict = pascal_voc_rw_ex.read_pascal_voc_xml(xml_path)
    width = int(xml_dict['annotation']['size']['width'])
    height = int(xml_dict['annotation']['size']['height'])
    if 'object' not in xml_dict['annotation'].keys():
        object_dict_list = []
    elif isinstance(xml_dict['annotation']['object'], list):
        object_dict_list = xml_dict['annotation']['object']
    else:
        object_dict_list = [xml_dict['annotation']['object']]

    filename = str.encode(os.path.basename(image_path))
    format = tf.cast('jpeg', tf.string)
    image_format = str.encode(format.numpy().decode('utf-8'))

    x_min_array = tf.TensorArray(tf.float32, size=len(object_dict_list))
    x_max_array = tf.TensorArray(tf.float32, size=len(object_dict_list))
    y_min_array = tf.TensorArray(tf.float32, size=len(object_dict_list))
    y_max_array = tf.TensorArray(tf.float32, size=len(object_dict_list))
    class_label_array = tf.TensorArray(tf.string, size=len(object_dict_list))
    class_index_array = tf.TensorArray(tf.int64, size=len(object_dict_list))

    for index, object_dict in enumerate(object_dict_list):
        x_min_array = x_min_array.write(index, tf.cast(int(object_dict['bndbox']['xmin']) / width, tf.float32))
        x_max_array = x_max_array.write(index, tf.cast(int(object_dict['bndbox']['xmax']) / width, tf.float32))
        y_min_array = y_min_array.write(index, tf.cast(int(object_dict['bndbox']['ymin']) / height, tf.float32))
        y_max_array = y_max_array.write(index, tf.cast(int(object_dict['bndbox']['ymax']) / height, tf.float32))
        class_label_array = class_label_array.write(index, tf.cast(object_dict['name'], tf.string))
        class_index_array = class_index_array.write(index, tf.cast(classes.index(object_dict['name']), tf.int64))

    classes_text = [str.encode(label.decode('utf-8')) for label in class_label_array.stack().numpy().tolist()]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(x_min_array.stack().numpy().ravel().tolist()),
        'image/object/bbox/xmax': float_list_feature(x_max_array.stack().numpy().ravel().tolist()),
        'image/object/bbox/ymin': float_list_feature(y_min_array.stack().numpy().ravel().tolist()),
        'image/object/bbox/ymax': float_list_feature(y_max_array.stack().numpy().ravel().tolist()),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(class_index_array.stack()),
    }))
    return tf_example
