def create_example(image, label):
    features = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _int64_feature(label),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'channels': _int64_feature(image.shape[2])
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def create_TFRecord(files_per_shard, images, labels, index_of_record):
    shard = len(images) // files_per_shard + int((len(images))%files_per_shard != 0)
    index = 0

    for i in range(shard):
        files_to_write = min(files_per_shard, len(images) - i*files_per_shard)
        with tf.io.TFRecordWriter(f'tfrecords/record_{index_of_record}.tfrecord') as writer:
            for j in range(files_to_write):
                example = create_example(images[index], labels[index])
                writer.write(example)
                index += 1
