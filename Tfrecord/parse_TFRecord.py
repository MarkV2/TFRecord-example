def parse_TFRecord_element(element):
    features = {
        'image' : tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.int64),
        'height':tf.io.FixedLenFeature([], tf.int64),
        'width':tf.io.FixedLenFeature([], tf.int64),
        'channels':tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, features)

    image = content['image']
    label = content['label']
    height = content['height']
    width = content['width']
    channels = content['channels']

    image = tf.io.parse_tensor(image, out_type=tf.float32)
    image = tf.reshape(image, shape=[height,width,channels])
    return (image, int(label))


def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        filenames
    )
    dataset = dataset.with_options(
        ignore_order
    )
    dataset = dataset.map(
        parse_TFRecord_element, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(8500)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(256)
    return dataset

train_records = [f'tfrecords/record_{i}.tfrecord' for i in range(0, 300)] # <-- this range depends on number of TFRecords you have
validation_records = [f'tfrecords/record_{i}.tfrecord' for i in range(300, 303)]
test_records = [f'tfrecords/record_{i}.tfrecord' for i in range(303, 306)]

train_dataset = get_dataset(train_records)
validation_dataset = get_dataset(validation_records)
test_dataset = get_dataset(test_records)
