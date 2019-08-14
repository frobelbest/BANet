import os
import sys
import importlib.util
import tensorflow as tf
import numpy as np

_loaded_module_id = 0


def load_myNetworks_module(module_name, path_to_myNetworks):
    """Returns the myNetworks module

    module_name: str
        Module name

    path_to_myNetworks: str
        Path to the 'myNetworks.py' inside the 'net' module folder, which contains the myNetworks.py
    """
    module_path = os.path.dirname(path_to_myNetworks)
    myNetworks_name = os.path.splitext(os.path.split(path_to_myNetworks)[1])[0]

    # add the __init__.py to the path if necessary
    module_path = os.path.join(module_path,'__init__.py')
        
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # this allows using importlib.import_module('.submod', module_name)
    sys.modules[module_name] = module
    
    return importlib.import_module('.'+myNetworks_name, module_name)


def load_myNetworks_module_noname(path_to_myNetworks):
    """Returns the myNetworks module

    path_to_myNetworks: str
        Path to the 'myNetworks.py' inside the 'net' module folder, which contains the myNetworks.py
    """
    global _loaded_module_id
    module_name = '___myNetworks_module_{0}'.format(_loaded_module_id)
    _loaded_module_id += 1
    return load_myNetworks_module(module_name, path_to_myNetworks)
   

# function based on https://github.com/tensorflow/tensorflow/issues/312
def optimistic_restore(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    """This function tries to restore all variables in the save file.

    This function ignores variables that do not exist or have incompatible shape.
    Raises TypeError if the there is a type mismatch for compatible shapes.
    
    session: tf.Session
        The tf session

    save_file: str
        Path to the checkpoint without the .index, .meta or .data extensions.

    ignore_vars: list, tuple or set of str
        These variables will be ignored.

    verbose: bool
        If True prints which variables will be restored

    ignore_incompatible_shapes: bool
        If True ignores variables with incompatible shapes.
        If False raises a runtime error f shapes are incompatible.

    """
    def vprint(*args, **kwargs): 
        if verbose: print(*args, flush=True, **kwargs)
    # def dbg(*args, **kwargs): print(*args, flush=True, **kwargs)
    def dbg(*args, **kwargs): pass
    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes and not var.name.split(':')[0] in ignore_vars])
    restore_vars = []

    dbg(saved_shapes)
    for v in tf.global_variables():
        dbg(v)

    nonfinite_values = False

    with tf.variable_scope('', reuse=True):
        for var_name, var_dtype, saved_var_name in var_names:
            dbg( var_name, var_dtype, saved_var_name, end='')
            curr_var = tf.get_variable(saved_var_name, dtype=var_dtype)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                dbg( ' shape OK')
                tmp = reader.get_tensor(saved_var_name)
                dbg(tmp.dtype)

                # check if there are nonfinite values in the tensor
                if not np.all(np.isfinite(tmp)):
                    nonfinite_values = True
                    print('{0} contains nonfinite values!'.format(saved_var_name), flush=True)

                if isinstance(tmp, np.ndarray):
                    saved_dtype = tf.as_dtype(tmp.dtype)
                else:
                    saved_dtype = tf.as_dtype(type(tmp))
                dbg(saved_dtype, var_dtype, saved_dtype.is_compatible_with(var_dtype))
                if not saved_dtype.is_compatible_with(var_dtype):
                    raise TypeError('types are not compatible for {0}: saved type {1}, variable type {2}.'.format(
                        saved_var_name, saved_dtype.name, var_dtype.name))

                vprint('restoring    ', saved_var_name)
                restore_vars.append(curr_var)
            else:
                vprint('not restoring',saved_var_name, 'incompatible shape:', var_shape, 'vs', saved_shapes[saved_var_name])
                if not ignore_incompatible_shapes:
                    raise RuntimeError('failed to restore "{0}" because of incompatible shapes: var: {1} vs saved: {2} '.format(saved_var_name, var_shape, saved_shapes[saved_var_name]))

    if nonfinite_values:
        raise RuntimeError('"{0}" contains nonfinite values!'.format(save_file))

    dbg( '-1-')
    saver = tf.train.Saver(
            var_list=restore_vars,
            restore_sequentially=True,)
    dbg( '-2-')
    saver.restore(session, save_file)
    dbg( '-3-')

