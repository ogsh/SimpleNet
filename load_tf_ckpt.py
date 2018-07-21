import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as ckpt
from tensorflow.python import pywrap_tensorflow

#def load_tf_ckpt(model):



model_path = "../tensorflow-yolo-v3/export/model.ckpt"

#ckpt.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)

all_tensors = all_tensor_names = True
reader = pywrap_tensorflow.NewCheckpointReader(model_path)
if all_tensors or all_tensor_names:
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)

