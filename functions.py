import tensorflow as tf

def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    intersect = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), tf.float32))
    
    iou = tf.reduce_sum(intersect) / (tf.reduce_sum(union) + tf.reduce_sum(intersect))
    return iou

