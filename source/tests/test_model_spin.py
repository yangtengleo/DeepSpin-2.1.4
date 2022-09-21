import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data, gen_data, del_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModelSpin(tf.test.TestCase):
    def setUp(self) :
        gen_data()

    def tearDown(self):
        del_data()

    def test_model_spin(self):        
        jfile = 'NiO_spin.json'
        jdata = j_loader(jfile)

        # set system information
        systems = j_must_have(jdata, 'systems')
        set_pfx = j_must_have(jdata, 'set_prefix')
        batch_size = j_must_have(jdata, 'batch_size')
        test_size = j_must_have(jdata, 'numb_test')
        stop_batch = j_must_have(jdata, 'stop_batch')
        rcut = j_must_have(jdata['model']['descriptor'], 'rcut')
        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)        
        test_data = data.get_test()

        # initialize model
        jdata['model']['descriptor'].pop('type', None)        
        descrpt = DescrptSeA(**jdata['model']['descriptor'], **jdata['model']['spin'], uniform_seed=True)
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = EnerFitting(**jdata['model']['fitting_net'], uniform_seed=True)
        spin = jdata['model']['spin']
        model = EnerModel(descrpt, fitting, spin=spin)

        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']]
        }

        model._compute_input_stat(input_data)
        model.descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c           = tf.placeholder(tf.float32,                  [5],        name='t_prop_c')
        t_energy           = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None],     name='t_energy')
        t_coord            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION,   [None],     name='i_coord')
        t_type             = tf.placeholder(tf.int32,                    [None],     name='i_type')
        t_natoms           = tf.placeholder(tf.int32,                    [model.ntypes+2], name='i_natoms')
        t_box              = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION,   [None, 9],  name='i_box')
        t_mesh             = tf.placeholder(tf.int32,                    [None],     name='i_mesh')
        is_training        = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred = model.build(t_coord, 
                                 t_type, 
                                 t_natoms, 
                                 t_box, 
                                 t_mesh,
                                 t_fparam,
                                 suffix = "model_spin", 
                                 reuse = False)
        energy = model_pred['energy']
        force  = model_pred['force']
        virial = model_pred['virial']

        # feed data and get results
        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_energy:        test_data['energy'] [:test_size],
                          t_coord:         test_data['coord']  [:test_size, :],
                          t_box:           test_data['box']    [:test_size, :],
                          t_type:          test_data['type'],
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False
        }
        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [out_ener, out_force, out_virial] = sess.run([energy, force, virial], 
                                                      feed_dict = feed_dict_test)

        natoms_real = np.sum(test_data['natoms_vec'][2 : 2 + len(spin['use_spin'])])
        force_real = np.reshape(out_force[:, :natoms_real * 3], [-1, 3])
        force_mag = np.reshape(out_force[:, natoms_real * 3:], [-1, 3])
        print(force_real[:5, :])
        print(force_mag[:5, :])


if __name__ == '__main__':
    unittest.main()