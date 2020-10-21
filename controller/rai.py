from controller.constants import INPUT_SHAPE, MODELS_REPO
from tensorflow.keras.applications.vgg16 import VGG16
import pickle, os, redisai
from os import listdir
from os.path import splitext
import tensorflow as tf

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants


def load_models_to_redisai(device="CPU"):
    rai = redisai.Client(host="redis", port="6379")
    try:
        with open(MODELS_REPO + "vgg16.pb", "rb") as f:
            model = f.read()
    except:
        model = VGG16(include_top=False, input_shape=INPUT_SHAPE)
        ins = [node.op.name for node in model.inputs]
        outs = [node.op.name for node in model.outputs]
        with open(MODELS_REPO + "vgg16.pkl", "wb") as cp_file:
            pickle.dump(dict(inputs=ins, outputs=outs), cp_file)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            frozen_graph = var_converter(sess, sess.graph_def, outs)
        tf.train.write_graph(frozen_graph, MODELS_REPO, "vgg16.pb", as_text=False)
    names = list(set(splitext(file)[0] for file in listdir(MODELS_REPO)))
    for name in names:
        with open(MODELS_REPO + f"{name}.pkl", "rb") as f:
            cp = pickle.load(f)
            inputs = cp["inputs"]
            outputs = cp["outputs"]
        with open(MODELS_REPO + f"{name}.pb", "rb") as f:
            model = f.read()
        rai.modelset(name, "TF", device, inputs=inputs, outputs=outputs, data=model)
