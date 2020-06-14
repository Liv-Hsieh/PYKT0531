import tensorflow as tf


@tf.function
def add(p, q):
    return tf.add(p, q) ** p + tf.add(p, q)


# 手動建目錄logs
LOG_DIR = 'logs/demo50'
writer = tf.summary.create_file_writer(LOG_DIR)

tf.summary.trace_on(graph=True, profiler=True)
v1 = add([1, 2, 3], [4, 5, 6])
print(v1)
with writer.as_default():
    tf.summary.trace_export(name="my add function", step=0,
                            profiler_outdir=LOG_DIR)