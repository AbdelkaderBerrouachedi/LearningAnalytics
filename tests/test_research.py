import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import research as r

class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=r.load_data(Path(__file__).resolve().parents[1]/'data/demo_raw')
    def copy_data(self):return {k:v.copy() for k,v in self.d.items()}
    def test_registration_boundary(self):
        d=self.copy_data();d['base']=d['base'].iloc[:3].copy();d['base']['date_unregistration']=np.nan
        d['base']['date_registration']=[27,28,29]
        self.assertEqual(len(r.snapshot(d,28)),2)
    def test_missing_registration_excluded_primary(self):
        d=self.copy_data();d['base']=d['base'].iloc[:1].copy();d['base']['date_unregistration']=np.nan;d['base']['date_registration']=np.nan
        self.assertEqual(len(r.snapshot(d,28)),0)
        self.assertEqual(len(r.snapshot(d,28,True)),1)
    def test_withdrawal_boundary_and_end(self):
        d=self.copy_data();d['base']=d['base'].iloc[:4].copy();d['base']['date_registration']=-1
        d['base']['final_result']='Withdrawn';d['base']['date_unregistration']=[27,28,29,1000]
        out=r.snapshot(d,28);self.assertEqual(len(out),1);self.assertEqual(out.y_withdrawal.iloc[0],1)
    def test_postcutoff_invariance(self):
        d=self.copy_data();original=r.snapshot(d,28)
        new=d['studentVle'].iloc[:1].copy();new['date']=29;new['sum_click']=999999
        d['studentVle']=pd.concat([d['studentVle'],new]);new=d['studentAssessment'].iloc[:1].copy();new['date_submitted']=29;new['score']=100
        d['studentAssessment']=pd.concat([d['studentAssessment'],new])
        pd.testing.assert_frame_equal(original,r.snapshot(d,28))
    def test_seven_day_window(self):
        d=self.copy_data();s=d['base'].iloc[0];d['base']=d['base'].iloc[:1].copy()
        d['base']['date_unregistration']=np.nan;d['base']['date_registration']=-1
        v=d['studentVle'].iloc[:2].copy()
        for k in r.KEY:v[k]=s[k]
        v['date']=[21,22];v['sum_click']=[100,3]
        d['studentVle']=v;out=r.snapshot(d,28)
        self.assertEqual(out.clicks_last_7d.iloc[0],3)
        self.assertEqual(out.clicks_last_14d.iloc[0],103)
    def test_constant_scores_expected_budget(self):
        p=np.ones(100);y=np.r_[np.ones(10),np.zeros(90)]
        w=r.budget_weights(p,.1);self.assertAlmostEqual(w.sum(),10)
        self.assertAlmostEqual(w@y/w.sum(),.1)
    def test_budget_permutation(self):
        rng=np.random.default_rng(3);p=np.r_[np.ones(50),np.zeros(50)];y=rng.integers(0,2,100);ix=rng.permutation(100)
        self.assertAlmostEqual(r.budget_weights(p,.1)@y,r.budget_weights(p[ix],.1)@y[ix])
    def test_exact_tied_budget(self):
        w=r.budget_weights([.9,.8,.8,.8,.1],.4)
        np.testing.assert_allclose(w,[1,1/3,1/3,1/3,0]);self.assertAlmostEqual(w.sum(),2)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):r.metrics([0,1],[.1,np.nan])
    def test_degenerate_auroc_undefined(self):self.assertIsNone(r.metrics([0,0],[.1,.2])['auroc'])
    def test_metrics_known(self):
        m=r.metrics([0,1,0,1],[.1,.9,.2,.8]);self.assertEqual(m['ap'],1);self.assertEqual(m['auroc'],1)
    def test_paired_identical_zero(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.9,.2,.8]);o=r.bootstrap(y,p,[1,1,2,2],20,other=p)
        self.assertEqual(o['ap_low'],0);self.assertEqual(o['ap_high'],0)
    def test_train_only_imputation(self):
        from sklearn.impute import SimpleImputer
        x=np.array([[1],[3],[np.nan]]);imp=SimpleImputer().fit(x)
        self.assertEqual(imp.transform([[np.nan],[1000]])[0,0],2)
    def test_primary_scores_absent(self):self.assertNotIn('mean_assessment_score',r.FEATURES)
    def test_feature_count(self):self.assertEqual(len(r.FEATURES),36);self.assertEqual(len(set(r.FEATURES)),36)
    def test_demo_cannot_use_real_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
            with self.assertRaises(ValueError):r.run('missing',Path(td)/'real',mode='demo',n_boot=1)

if __name__=='__main__':unittest.main(verbosity=2)

