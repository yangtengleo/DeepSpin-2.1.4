#!/usr/bin/env python3
"""Script for freezing TF trained graph so it can be used with LAMMPS and i-PI.

References
----------
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
"""

import logging
import google.protobuf.message
from deepmd.env import tf, FITTING_NET_PATTERN
from deepmd.utils.errors import GraphTooLargeError
from deepmd.utils.sess import run_sess
from deepmd.utils.graph import get_pattern_nodes_from_graph_def
from os.path import abspath

# load grad of force module
import deepmd.op

from typing import List, Optional

from deepmd.nvnmd.entrypoints.freeze import save_weight

__all__ = ["freeze"]

log = logging.getLogger(__name__)

def _transfer_fitting_net_trainable_variables(sess, old_graph_def, raw_graph_def):
    old_pattern = FITTING_NET_PATTERN
    raw_pattern = FITTING_NET_PATTERN\
        .replace('idt',    'idt+_\d+')\
        .replace('bias',   'bias+_\d+')\
        .replace('matrix', 'matrix+_\d+')
    old_graph_nodes = get_pattern_nodes_from_graph_def(
        old_graph_def, 
        old_pattern
    )
    try :
        raw_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            raw_graph_def,  # The graph_def is used to retrieve the nodes
            [n + '_1' for n in old_graph_nodes],  # The output node names are used to select the usefull nodes
        )
    except AssertionError:
        # if there's no additional nodes
        return old_graph_def

    raw_graph_nodes = get_pattern_nodes_from_graph_def(
        raw_graph_def, 
        raw_pattern
    )
    for node in old_graph_def.node:
        if node.name not in old_graph_nodes.keys():
            continue
        tensor = tf.make_ndarray(raw_graph_nodes[node.name + '_1'])
        node.attr["value"].tensor.tensor_content = tensor.tostring()
    return old_graph_def

def _make_node_names(model_type: str, modifier_type: Optional[str] = None) -> List[str]:
    """Get node names based on model type.

    Parameters
    ----------
    model_type : str
        str type of model
    modifier_type : Optional[str], optional
        modifier type if any, by default None

    Returns
    -------
    List[str]
        list with all node names to freeze

    Raises
    ------
    RuntimeError
        if unknown model type
    """
    nodes = [
        "model_type",
        "descrpt_attr/rcut",
        "descrpt_attr/ntypes",
        "model_attr/tmap",
        "model_attr/model_type",
        "model_attr/model_version",
        "train_attr/min_nbor_dist",
        "train_attr/training_script",
    ]

    if model_type == "ener":
        nodes += [
            "o_energy",
            "o_force",
            "o_virial",
            "o_atom_energy",
            "o_atom_virial",
            "descrpt_attr/ntypes_spin",
            "fitting_attr/dfparam",
            "fitting_attr/daparam",
        ]
    elif model_type == "wfc":
        nodes += [
            "o_wfc",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "dipole":
        nodes += [
            "o_dipole",
            "o_global_dipole",
            "o_force",
            "o_virial",
            "o_atom_virial",
            "o_rmat",
            "o_rmat_deriv",
            "o_nlist",
            "o_rij",
            "descrpt_attr/sel",
            "descrpt_attr/ndescrpt",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "polar":
        nodes += [
            "o_polar",
            "o_global_polar",
            "o_force",
            "o_virial",
            "o_atom_virial",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "global_polar":
        nodes += [
            "o_global_polar",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    else:
        raise RuntimeError(f"unknow model type {model_type}")
    if modifier_type == "dipole_charge":
        nodes += [
            "modifier_attr/type",
            "modifier_attr/mdl_name",
            "modifier_attr/mdl_charge_map",
            "modifier_attr/sys_charge_map",
            "modifier_attr/ewald_h",
            "modifier_attr/ewald_beta",
            "dipole_charge/model_type",
            "dipole_charge/descrpt_attr/rcut",
            "dipole_charge/descrpt_attr/ntypes",
            "dipole_charge/model_attr/tmap",
            "dipole_charge/model_attr/model_type",
            "dipole_charge/model_attr/model_version",
            "o_dm_force",
            "dipole_charge/model_attr/sel_type",
            "dipole_charge/o_dipole",
            "dipole_charge/model_attr/output_dim",
            "o_dm_virial",
            "o_dm_av",
        ]
    return nodes


def freeze(
    *, checkpoint_folder: str, output: str, node_names: Optional[str] = None, nvnmd_weight: Optional[str] = None, **kwargs
):
    """Freeze the graph in supplied folder.

    Parameters
    ----------
    checkpoint_folder : str
        location of the folder with model
    output : str
        output file name
    node_names : Optional[str], optional
        names of nodes to output, by default None
    """
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(checkpoint_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # expand the output file to full path
    output_graph = abspath(output)

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep
    # and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    # node_names = "energy_test,force_test,virial_test,t_rcut"

    # We clear devices to allow TensorFlow to control
    # on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    try:
        # In case paralle training
        import horovod.tensorflow as _
    except ImportError:
        pass
    saver = tf.train.import_meta_graph(
        f"{input_checkpoint}.meta", clear_devices=clear_devices
    )

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    try:
        input_graph_def = graph.as_graph_def()
    except google.protobuf.message.DecodeError as e:
        raise GraphTooLargeError(
            "The graph size exceeds 2 GB, the hard limitation of protobuf."
            " Then a DecodeError was raised by protobuf. You should "
            "reduce the size of your model."
        ) from e
    nodes = [n.name for n in input_graph_def.node]

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        model_type = run_sess(sess, "model_attr/model_type:0", feed_dict={}).decode("utf-8")
        if "modifier_attr/type" in nodes:
            modifier_type = run_sess(sess, "modifier_attr/type:0", feed_dict={}).decode(
                "utf-8"
            )
        else:
            modifier_type = None
        if node_names is None:
            output_node_list = _make_node_names(model_type, modifier_type)
            different_set = set(output_node_list) - set(nodes)
            if different_set:
                log.warning(
                    "The following nodes are not in the graph: %s. "
                    "Skip freezeing these nodes. You may be freezing "
                    "a checkpoint generated by an old version." % different_set
                )
                # use intersection as output list
                output_node_list = list(set(output_node_list) & set(nodes))
        else:
            output_node_list = node_names.split(",")
        log.info(f"The following nodes will be frozen: {output_node_list}")

        if nvnmd_weight is not None:
            save_weight(sess, nvnmd_weight) # nvnmd

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_list,  # The output node names are used to select the usefull nodes
        )

        # If we need to transfer the fitting net variables
        output_graph_def = _transfer_fitting_net_trainable_variables(
            sess,
            output_graph_def,
            input_graph_def
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        log.info(f"{len(output_graph_def.node):d} ops in the final graph.")
