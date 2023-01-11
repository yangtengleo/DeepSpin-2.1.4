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
        jfile = 'model_spin/test_model_spin.json'
        jdata = j_loader(jfile)

        # set system information
        systems = j_must_have(jdata['training']['training_data'], 'systems')
        set_pfx = j_must_have(jdata['training'], 'set_prefix')
        batch_size = j_must_have(jdata['training']['training_data'], 'batch_size')
        test_size = j_must_have(jdata['training']['validation_data'], 'numb_btch')
        stop_batch = j_must_have(jdata['training'], 'numb_steps')
        rcut = j_must_have(jdata['model']['descriptor'], 'rcut')
        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)        
        test_data = data.get_test()

        # initialize model
        descrpt_param = jdata['model']['descriptor']
        spin_param = jdata['model']['spin']
        fitting_param = jdata['model']['fitting_net']
        descrpt = DescrptSeA(**descrpt_param, **spin_param, uniform_seed=True)
        fitting_param.pop('type', None)
        fitting_param['descrpt'] = descrpt
        fitting = EnerFitting(**fitting_param, uniform_seed=True)
        model = EnerModel(descrpt, fitting, spin=spin_param)
        
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
                          t_energy:        test_data['energy'],
                          t_coord:         np.reshape(test_data['coord']  ,  [-1]),
                          t_box:           np.reshape(test_data['box']    ,  [-1, 9]),
                          t_type:          np.reshape(test_data['type'],     [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False
        }

        print("t_natoms")
        print(feed_dict_test[t_natoms])
        print("t_type")
        print(feed_dict_test[t_type])

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [out_ener, out_force, out_virial] = sess.run([energy, force, virial], 
                                                      feed_dict = feed_dict_test)

        natoms_real = np.sum(test_data['natoms_vec'][2 : 2 + len(spin_param['use_spin'])])
        force_real = np.reshape(out_force[:, :natoms_real * 3], [-1, 3])
        force_mag = np.reshape(out_force[:, natoms_real * 3:], [-1, 3])
        print(force_real[:5, :])
        print(force_mag[:5, :])
        # print(out_ener[:5, :])
<<<<<<< HEAD
        direc = ['x', 'y', 'z']
        # 原子力 对比.
        for idx in range(6):
            print(f'解析原子力F_{idx//3}{direc[idx%3]}: ', force_real[idx//3, idx%3])
            print(f'数值原子力F_{idx//3}{direc[idx%3]}: ', -(out_ener[2*idx + 1] - out_ener[2*idx + 2]) / 0.02)
        # 磁性力 对比.
        for idx in range(6):
            print(f'解析磁性力F_{idx//3}{direc[idx%3]}: ', force_mag[idx//3, idx%3])
            print(f'数值磁性力F_{idx//3}{direc[idx%3]}: ', -(out_ener[2*idx + 13] - out_ener[2*idx + 14]) / 0.02)
=======
        # 原子力 对比.
        for idx in range(6):
            print(f'解析力F_{idx//3}_{idx%3}: ', force_real[idx//3, idx%3])
            print(f'数值力F_{idx//3}_{idx%3}: ', -(out_ener[2*idx + 1] - out_ener[2*idx + 2]) / 0.02)
        # 磁性力 对比.
        for idx in range(6):
            print(f'解析磁性力F_{idx//3}_{idx%3}: ', force_mag[idx//3, idx%3])
            print(f'数值磁性力F_{idx//3}_{idx%3}: ', -(out_ener[2*idx + 13] - out_ener[2*idx + 14]) / 0.02)
>>>>>>> 7acfbecdc0b545ac0426d41c0235c4131d97187b



if __name__ == '__main__':
    unittest.main()
