"""Adversarial tests for acceptance, not tests mirroring the implementation."""
import unittest
import numpy as np
from runtime_bootstrap import setup
setup()
from geometry_spec import Corridor, GateTracker
from metrics import metrics
from baselines.motion_overlap import plan


class AcceptanceTests(unittest.TestCase):
    def test_resolution_switch_is_reentrant(self):
        from execution_settings import configure_dt
        from compact_farvector_runtime import FarVectorRuntime
        try:
            for dt in [.0005,.001,.0005,.001]:
                configure_dt(dt)
                env=FarVectorRuntime(np.array([[0.,0.],[20.,0.]]))
                rows=env.step([0.,0.,0.])[3]['rows']
                self.assertEqual(len(rows),round(.02/dt))
        finally:
            configure_dt(.001)
    def test_command_endpoints_do_not_certify_interior(self):
        c=Corridor(np.array([[0.,0.],[0.,4.],[4.,4.]]),epsilon=.5)
        self.assertLess(c.point_distance([0,0]),.5)
        self.assertLess(c.point_distance([4,4]),.5)
        self.assertFalse(c.segment_inside([0,0],[4,4]))
    def test_closed_endpoint_is_not_complete(self):
        ref=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.],[0.,0.]])
        p=np.repeat(ref[:1],30,axis=0)
        r=metrics(ref,p)
        self.assertFalse(r['success']);self.assertFalse(r['geometry']['ordered_complete'])
    def test_vector_limits_not_individual_axes(self):
        p=np.c_[np.arange(30)*.09,np.arange(30)*.09]
        r=metrics(np.array([p[0],p[-1]]),p)
        self.assertGreater(r['vector_peaks'][0],100)
        self.assertFalse(r['kinematic_pass'])
    def test_startup_acceleration_is_included(self):
        p=np.c_[np.arange(30)*.05,np.zeros(30)]
        r=metrics(np.array([p[0],p[-1]]),p)
        self.assertGreater(r['vector_peaks'][1],2000)
        self.assertFalse(r['success'])
    def test_straight_scurve_accepts(self):
        ref=np.array([[0.,0.],[20.,0.]])
        result=plan(ref)
        r=metrics(ref,result['positions'])
        self.assertTrue(r['success'],r)
        self.assertGreaterEqual(r['terminal_rest_s'],.02)
    def test_zero_length_and_reversal_fail_explicitly(self):
        with self.assertRaises(ValueError):Corridor([[0,0],[0,0]])
        c=Corridor([[0,0],[1,0],[0,0]])
        with self.assertRaises(ValueError):GateTracker(c,[0,0])


if __name__=='__main__':unittest.main(verbosity=2)
