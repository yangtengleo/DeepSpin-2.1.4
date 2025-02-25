import warnings
import numpy as np
from typing import Tuple, List
from packaging.version import Version

from deepmd.env import tf
from deepmd.common import add_data_requirement, get_activation_func, get_precision, cast_precision
from deepmd.utils.network import one_layer_rand_seed_shift
from deepmd.utils.network import one_layer as one_layer_deepmd
from deepmd.utils.type_embed import embed_atom_type
from deepmd.utils.graph import get_fitting_net_variables_from_graph_def, load_graph_def, get_tensor_by_name_from_graph
from deepmd.fit.fitting import Fitting

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, TF_VERSION

from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.fit.ener import one_layer_nvnmd

class EnerFitting (Fitting):
    r"""Fitting the energy of the system. The force and the virial can also be trained.

    The potential energy :math:`E` is a fitting network function of the descriptor :math:`\mathcal{D}`:

    .. math::
        E(\mathcal{D}) = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)} \circ \mathcal{L}^{(0)}

    The first :math:`n` hidden layers :math:`\mathcal{L}^{(0)}, \cdots, \mathcal{L}^{(n-1)}` are given by

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b})

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}`$` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}`$` are weights and biases, respectively,
    both of which are trainable if `trainable[i]` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    The output layer :math:`\mathcal{L}^{(n)}` is given by

    .. math::
        \mathbf{y}=\mathcal{L}^{(n)}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \mathbf{x}^T\mathbf{w}+\mathbf{b}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_{n-1}}`$` is the input vector and :math:`\mathbf{y} \in \mathbb{R}`
    is the output scalar. :math:`\mathbf{w} \in \mathbb{R}^{N_{n-1}}` and
    :math:`\mathbf{b} \in \mathbb{R}`$` are weights and bias, respectively,
    both of which are trainable if `trainable[n]` is `True`.

    Parameters
    ----------
    descrpt
            The descrptor :math:`\mathcal{D}`
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    resnet_dt
            Time-step `dt` in the resnet construction:
            :math:`y = x + dt * \phi (Wx + b)`
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    rcond
            The condition number for the regression of atomic energy.
    tot_ener_zero
            Force the total energy to zero. Useful for the charge fitting.
    trainable
            If the weights of fitting net are trainable. 
            Suppose that we have :math:`N_l` hidden layers in the fitting net, 
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    seed
            Random seed for initializing the network parameters.
    atom_ener
            Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    """
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  numb_fparam : int = 0,
                  numb_aparam : int = 0,
                  rcond : float = 1e-3,
                  tot_ener_zero : bool = False,
                  trainable : List[bool] = None,
                  seed : int = None,
                  atom_ener : List[float] = [],
                  activation_function : str = 'tanh',
                  precision : str = 'default',
                  uniform_seed: bool = False,
                  use_spin: List[bool] = None,
                  spin_norm: List[float] = None,
                  virtual_len: List[float] = None
    ) -> None:
        """
        Constructor
        """
        # model param
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        # args = ()\
        #        .add('numb_fparam',      int,    default = 0)\
        #        .add('numb_aparam',      int,    default = 0)\
        #        .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
        #        .add('resnet_dt',        bool,   default = True)\
        #        .add('rcond',            float,  default = 1e-3) \
        #        .add('tot_ener_zero',    bool,   default = False) \
        #        .add('seed',             int)               \
        #        .add('atom_ener',        list,   default = [])\
        #        .add("activation_function", str,    default = "tanh")\
        #        .add("precision",           str, default = "default")\
        #        .add("trainable",        [list, bool], default = True)
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.rcond = rcond
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.use_spin = use_spin
        self.spin_norm = spin_norm
        self.virtual_len = virtual_len
        self.ntypes_spin = descrpt.get_ntypes_spin()
        self.seed_shift = one_layer_rand_seed_shift()
        self.tot_ener_zero = tot_ener_zero
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.trainable = trainable
        if self.trainable is None:
            self.trainable = [True for ii in range(len(self.n_neuron) + 1)]
        if type(self.trainable) is bool:
            self.trainable = [self.trainable] * (len(self.n_neuron)+1)
        assert(len(self.trainable) == len(self.n_neuron) + 1), 'length of trainable should be that of n_neuron + 1'
        self.atom_ener = []
        self.atom_ener_v = atom_ener
        for at, ae in enumerate(atom_ener):
            if ae is not None:
                self.atom_ener.append(tf.constant(ae, self.fitting_precision, name = "atom_%d_ener" % at))
            else:
                self.atom_ener.append(None)
        self.useBN = False
        self.bias_atom_e = np.zeros(self.ntypes, dtype=np.float64)
        # data requirement
        if self.numb_fparam > 0 :
            add_data_requirement('fparam', self.numb_fparam, atomic=False, must=True, high_prec=False)
            self.fparam_avg = None
            self.fparam_std = None
            self.fparam_inv_std = None
        if self.numb_aparam > 0:
            add_data_requirement('aparam', self.numb_aparam, atomic=True,  must=True, high_prec=False)
            self.aparam_avg = None
            self.aparam_std = None
            self.aparam_inv_std = None

        self.fitting_net_variables = None
        self.mixed_prec = None

    def get_numb_fparam(self) -> int:
        """
        Get the number of frame parameters
        """
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """
        Get the number of atomic parameters
        """
        return self.numb_fparam

    def compute_output_stats(self, 
                             all_stat: dict
    ) -> None:
        """
        Compute the ouput statistics

        Parameters
        ----------
        all_stat
                must have the following components:
                all_stat['energy'] of shape n_sys x n_batch x n_frame
                can be prepared by model.make_stat_input
        """
        self.bias_atom_e = self._compute_output_stats(all_stat, rcond = self.rcond)

    def _compute_output_stats(self, all_stat, rcond = 1e-3):
        data = all_stat['energy']
        # data[sys_idx][batch_idx][frame_idx]
        sys_ener = np.array([])
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data)
            sys_ener = np.append(sys_ener, np.average(sys_data))
        data = all_stat['natoms_vec']
        sys_tynatom = np.array([])
        nsys = len(data)
        for ss in range(len(data)):
            sys_tynatom = np.append(sys_tynatom, data[ss][0].astype(np.float64))
        sys_tynatom = np.reshape(sys_tynatom, [nsys,-1])
        sys_tynatom = sys_tynatom[:,2:]
        if len(self.atom_ener) > 0:
            # Atomic energies stats are incorrect if atomic energies are assigned.
            # In this situation, we directly use these assigned energies instead of computing stats.
            # This will make the loss decrease quickly
            assigned_atom_ener = np.array(list((ee for ee in self.atom_ener_v if ee is not None)))
            assigned_ener_idx = list((ii for ii, ee in enumerate(self.atom_ener_v) if ee is not None))
            # np.dot out size: nframe
            sys_ener -= np.dot(sys_tynatom[:, assigned_ener_idx], assigned_atom_ener)
            sys_tynatom[:, assigned_ener_idx] = 0.
        energy_shift,resd,rank,s_value \
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = rcond)
        if len(self.atom_ener) > 0:
            for ii in assigned_ener_idx:
                energy_shift[ii] = self.atom_ener_v[ii]
        return energy_shift    

    def compute_input_stats(self, 
                            all_stat : dict,
                            protection : float = 1e-2) -> None:
        """
        Compute the input statistics

        Parameters
        ----------
        all_stat
                if numb_fparam > 0 must have all_stat['fparam']
                if numb_aparam > 0 must have all_stat['aparam']
                can be prepared by model.make_stat_input
        protection
                Divided-by-zero protection
        """
        # stat fparam
        if self.numb_fparam > 0:
            cat_data = np.concatenate(all_stat['fparam'], axis = 0)
            cat_data = np.reshape(cat_data, [-1, self.numb_fparam])
            self.fparam_avg = np.average(cat_data, axis = 0)
            self.fparam_std = np.std(cat_data, axis = 0)
            for ii in range(self.fparam_std.size):
                if self.fparam_std[ii] < protection:
                    self.fparam_std[ii] = protection
            self.fparam_inv_std = 1./self.fparam_std
        # stat aparam
        if self.numb_aparam > 0:
            sys_sumv = []
            sys_sumv2 = []
            sys_sumn = []
            for ss_ in all_stat['aparam'] : 
                ss = np.reshape(ss_, [-1, self.numb_aparam])
                sys_sumv.append(np.sum(ss, axis = 0))
                sys_sumv2.append(np.sum(np.multiply(ss, ss), axis = 0))
                sys_sumn.append(ss.shape[0])
            sumv = np.sum(sys_sumv, axis = 0)
            sumv2 = np.sum(sys_sumv2, axis = 0)
            sumn = np.sum(sys_sumn)
            self.aparam_avg = (sumv)/sumn
            self.aparam_std = self._compute_std(sumv2, sumv, sumn)
            for ii in range(self.aparam_std.size):
                if self.aparam_std[ii] < protection:
                    self.aparam_std[ii] = protection
            self.aparam_inv_std = 1./self.aparam_std


    def _compute_std (self, sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))

    def _build_lower(
            self,
            start_index,
            natoms,
            inputs,
            fparam = None,
            aparam = None, 
            bias_atom_e = 0.0,
            suffix = '',
            reuse = None
    ):
        # cut-out inputs
        inputs_i = tf.slice (inputs,
                             [ 0, start_index, 0],
                             [-1, natoms, -1] )
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        layer = inputs_i
        if fparam is not None:
            ext_fparam = tf.tile(fparam, [1, natoms])
            ext_fparam = tf.reshape(ext_fparam, [-1, self.numb_fparam])
            ext_fparam = tf.cast(ext_fparam,self.fitting_precision)
            layer = tf.concat([layer, ext_fparam], axis = 1)
        if aparam is not None:
            ext_aparam = tf.slice(aparam, 
                                  [ 0, start_index      * self.numb_aparam],
                                  [-1, natoms * self.numb_aparam])
            ext_aparam = tf.reshape(ext_aparam, [-1, self.numb_aparam])
            ext_aparam = tf.cast(ext_aparam,self.fitting_precision)
            layer = tf.concat([layer, ext_aparam], axis = 1)

        if nvnmd_cfg.enable: 
            one_layer = one_layer_nvnmd
        else:
            one_layer = one_layer_deepmd
        for ii in range(0,len(self.n_neuron)) :
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] and (not nvnmd_cfg.enable):
                layer+= one_layer(
                    layer,
                    self.n_neuron[ii],
                    name='layer_'+str(ii)+suffix,
                    reuse=reuse,
                    seed = self.seed,
                    use_timestep = self.resnet_dt,
                    activation_fn = self.fitting_activation_fn,
                    precision = self.fitting_precision,
                    trainable = self.trainable[ii],
                    uniform_seed = self.uniform_seed,
                    initial_variables = self.fitting_net_variables,
                    mixed_prec = self.mixed_prec)
            else :
                layer = one_layer(
                    layer,
                    self.n_neuron[ii],
                    name='layer_'+str(ii)+suffix,
                    reuse=reuse,
                    seed = self.seed,
                    activation_fn = self.fitting_activation_fn,
                    precision = self.fitting_precision,
                    trainable = self.trainable[ii],
                    uniform_seed = self.uniform_seed,
                    initial_variables = self.fitting_net_variables,
                    mixed_prec = self.mixed_prec)
            if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
        final_layer = one_layer(
            layer, 
            1, 
            activation_fn = None, 
            bavg = bias_atom_e, 
            name='final_layer'+suffix, 
            reuse=reuse, 
            seed = self.seed, 
            precision = self.fitting_precision, 
            trainable = self.trainable[-1],
            uniform_seed = self.uniform_seed,
            initial_variables = self.fitting_net_variables,
            mixed_prec = self.mixed_prec,
            final_layer = True)
        if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift

        return final_layer
            
            
    @cast_precision
    def build (self, 
               inputs : tf.Tensor,
               natoms : tf.Tensor,
               input_dict : dict = None,
               reuse : bool = None,
               suffix : str = '', 
    ) -> tf.Tensor:
        """
        Build the computational graph for fitting net

        Parameters
        ----------
        inputs
                The input descriptor
        input_dict
                Additional dict for inputs. 
                if numb_fparam > 0, should have input_dict['fparam']
                if numb_aparam > 0, should have input_dict['aparam']
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        ener
                The system energy
        """
        if input_dict is None:
            input_dict = {}
        bias_atom_e = self.bias_atom_e
        if self.numb_fparam > 0:
            if self.fparam_avg is None:
                self.fparam_avg = 0.
            if self.fparam_inv_std is None:
                self.fparam_inv_std = 1.
        if self.numb_aparam > 0:
            if self.aparam_avg is None:
                self.aparam_avg = 0.
            if self.aparam_inv_std is None:
                self.aparam_inv_std = 1.

        with tf.variable_scope('fitting_attr' + suffix, reuse = reuse) :
            t_dfparam = tf.constant(self.numb_fparam, 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
            t_daparam = tf.constant(self.numb_aparam, 
                                    name = 'daparam', 
                                    dtype = tf.int32)
            if self.numb_fparam > 0: 
                t_fparam_avg = tf.get_variable('t_fparam_avg', 
                                               self.numb_fparam,
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.fparam_avg))
                t_fparam_istd = tf.get_variable('t_fparam_istd', 
                                                self.numb_fparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.fparam_inv_std))
            if self.numb_aparam > 0: 
                t_aparam_avg = tf.get_variable('t_aparam_avg', 
                                               self.numb_aparam,
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.aparam_avg))
                t_aparam_istd = tf.get_variable('t_aparam_istd', 
                                                self.numb_aparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.aparam_inv_std))
            
        inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt])
        if len(self.atom_ener):
            # only for atom_ener
            nframes = input_dict.get('nframes')
            if nframes is not None:
                # like inputs, but we don't want to add a dependency on inputs
                inputs_zero = tf.zeros((nframes, natoms[0], self.dim_descrpt), dtype=self.fitting_precision)
            else:
                inputs_zero = tf.zeros_like(inputs, dtype=self.fitting_precision)
        

        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        fparam = None
        aparam = None
        if self.numb_fparam > 0 :
            fparam = input_dict['fparam']
            fparam = tf.reshape(fparam, [-1, self.numb_fparam])
            fparam = (fparam - t_fparam_avg) * t_fparam_istd            
        if self.numb_aparam > 0 :
            aparam = input_dict['aparam']
            aparam = tf.reshape(aparam, [-1, self.numb_aparam])
            aparam = (aparam - t_aparam_avg) * t_aparam_istd
            aparam = tf.reshape(aparam, [-1, self.numb_aparam * natoms[0]])
            
        type_embedding = input_dict.get('type_embedding', None)
        if type_embedding is not None:
            atype_embed = embed_atom_type(self.ntypes, natoms, type_embedding)
            atype_embed = tf.tile(atype_embed,[tf.shape(inputs)[0],1])
        else:
            atype_embed = None

        if atype_embed is None:
            start_index = 0
            outs_list = []
            ntypes_atom = self.ntypes - self.ntypes_spin
            for type_i in range(self.ntypes):
                if type_i >= ntypes_atom:
                    break 
                if bias_atom_e is None :
                    type_bias_ae = 0.0
                elif self.use_spin is None:
                    type_bias_ae = bias_atom_e[type_i]
                else :
                    if self.use_spin[type_i]:
                        type_bias_ae = bias_atom_e[type_i] + bias_atom_e[type_i + ntypes_atom]
                    else:
                        type_bias_ae = bias_atom_e[type_i]
                final_layer = self._build_lower(start_index, 
                                                natoms[2+type_i], 
                                                inputs, fparam, aparam, 
                                                bias_atom_e = type_bias_ae, 
                                                suffix = '_type_'+str(type_i)+suffix, 
                                                reuse = reuse)
                # concat the results
                if type_i < len(self.atom_ener) and self.atom_ener[type_i] is not None:                
                    zero_layer = self._build_lower(start_index, 
                                                   natoms[2+type_i], 
                                                   inputs_zero, fparam, aparam, 
                                                   bias_atom_e = type_bias_ae, 
                                                   suffix = '_type_'+str(type_i)+suffix, 
                                                   reuse = True)
                    final_layer += self.atom_ener[type_i] - zero_layer
                final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i]])
                outs_list.append(final_layer)
                start_index += natoms[2+type_i]
            # concat the results
            # concat once may be faster than multiple concat
            outs = tf.concat(outs_list, axis = 1)
        # with type embedding
        else:
            if len(self.atom_ener) > 0:
                raise RuntimeError("setting atom_ener is not supported by type embedding")
            atype_embed = tf.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat(
                [tf.reshape(inputs,[-1,self.dim_descrpt]),atype_embed],
                axis=1
            )
            self.dim_descrpt = self.dim_descrpt + type_shape[1]
            inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt])
            final_layer = self._build_lower(
                0, natoms[0], 
                inputs, fparam, aparam, 
                bias_atom_e=0.0, suffix=suffix, reuse=reuse
            )
            outs = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[0]])
            # add atom energy bias; TF will broadcast to all batches
            # tf.repeat is avaiable in TF>=2.1 or TF 1.15
            _TF_VERSION = Version(TF_VERSION)
            if (Version('1.15') <= _TF_VERSION < Version('2') or _TF_VERSION >= Version('2.1')) and self.bias_atom_e is not None:
                outs += tf.repeat(tf.Variable(self.bias_atom_e, dtype=self.fitting_precision, trainable=False, name="bias_atom_ei"), natoms[2:])

        if self.tot_ener_zero:
            force_tot_ener = 0.0
            outs = tf.reshape(outs, [-1, natoms[0]])
            outs_mean = tf.reshape(tf.reduce_mean(outs, axis = 1), [-1, 1])
            outs_mean = outs_mean - tf.ones_like(outs_mean, dtype = GLOBAL_TF_FLOAT_PRECISION) * (force_tot_ener/global_cvt_2_tf_float(natoms[0]))
            outs = outs - outs_mean
            outs = tf.reshape(outs, [-1])

        tf.summary.histogram('fitting_net_output', outs)
        return tf.reshape(outs, [-1])


    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the fitting net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.fitting_net_variables = get_fitting_net_variables_from_graph_def(graph_def)
        if self.numb_fparam > 0:
            self.fparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_avg' % suffix)
            self.fparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_istd' % suffix)
        if self.numb_aparam > 0:
            self.aparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_avg' % suffix)
            self.aparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_istd' % suffix)

    def enable_compression(self,
                           model_file: str,
                           suffix: str = ""
    ) -> None:
        """
        Set the fitting net attributes from the frozen model_file when fparam or aparam is not zero

        Parameters
        ----------
        model_file : str
            The input frozen model file
        suffix : str, optional
                The suffix of the scope
        """
        if self.numb_fparam > 0 or self.numb_aparam > 0:
            graph, _ = load_graph_def(model_file)
        if self.numb_fparam > 0:
            self.fparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_avg' % suffix)
            self.fparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_istd' % suffix)
        if self.numb_aparam > 0:
            self.aparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_avg' % suffix)
            self.aparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_istd' % suffix)
 

    def enable_mixed_precision(self, mixed_prec: dict = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec['output_prec'])
