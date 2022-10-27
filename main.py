import argparse
import os
import glob
from multiprocessing import Process, cpu_count
import tensorflow as tf
from tqdm import tqdm

from io_tfrecords import pascal_voc2tf_example

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def write(image_xml_path_list, output_dir_path, classes, proc_num, records_prefix_index):
    tf_record_writer = tf.io.TFRecordWriter(os.path.join(output_dir_path, f"dataset.tfrecords-{records_prefix_index}{proc_num:03d}"))
    for image_path, xml_path in tqdm(image_xml_path_list):
        example = pascal_voc2tf_example(image_path, xml_path, classes)
        tf_record_writer.write(example.SerializeToString())
    tf_record_writer.close()

def split(a_list, split_num):
    result_list = []
    for split_index in range(split_num):
        result_list.append([])
    for index, an_elem in enumerate(a_list):
        result_list[index % split_num].append(an_elem)
    return result_list

def main(input_image_dir_path, input_classes_path, output_dir_path, records_prefix_index, cpu_count):
    os.makedirs(output_dir_path, exist_ok=True)

    classes = []
    with open(input_classes_path) as f:
        for line in f:
            classes.append(line.strip())

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(input_image_dir_path, files), recursive=True))

    image_xml_path_list = []
    for image_path in image_path_list:
        xml_path = os.path.splitext(image_path)[0]+'.xml'
        if os.path.exists(xml_path):
            image_xml_path_list.append((image_path, xml_path))

    split_image_xml_path_list = split(image_xml_path_list, cpu_count)

    processes = []
    for proc_num in range(len(split_image_xml_path_list)):
        p = Process(target=write, args=(split_image_xml_path_list[proc_num], output_dir_path, classes, proc_num, records_prefix_index))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/train')
    parser.add_argument('--input_classes_path', type=str,
                        default='~/.vaik-mnist-detection-dataset/classes.txt')
    parser.add_argument('--output_dir_path', type=str,
                        default='~/.vaik-pascalvoc2tfrecord-mp/vaik-mnist-detection-dataset/train')
    parser.add_argument('--records_prefix_index', type=str, default='00')
    parser.add_argument('--cpu_count', type=int, default=cpu_count())
    args = parser.parse_args()

    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
