import tensorflow as tf


@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


LOG_DIR = "logs/demo52"
writer = tf.summary.create_file_writer(LOG_DIR)
tf.summary.trace_on(graph=True, profiler=True)
triangles = [[3, 4, 5], [6, 8, 10], [6, 6, 6], [2.3, 4.1, 4.8], [5, 3, 7]]
result = computeArea(tf.constant(triangles))
with writer.as_default():
    tf.summary.trace_export(name='my heron fomula', step=0, profiler_outdir=LOG_DIR)

print(result)