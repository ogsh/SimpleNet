import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as ckpt
import numpy as np

from collections import OrderedDict
#def load_tf_ckpt(model):


model_path = "../tensorflow-yolo-v3/export/model.ckpt"

#ckpt.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)

all_tensors = all_tensor_names = True
reader = tf.python.pywrap_tensorflow.NewCheckpointReader(model_path)

tf_dict = OrderedDict()

if all_tensors or all_tensor_names:
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)
    tf_dict[key] = np.array(reader.get_tensor(key))


