import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from final_benchmark_analysis import independent_metrics, bootstrap_difference

def record(case_id,flag,truth,category='Street Light'):
    return {'case_id':case_id,'prediction':{'misroute':flag,'recommended_category':category,'reason':'Unit fixture'},
            'truth':truth,'original':'Street Light','current':'Street Light',
            'usage':{'input_tokens':10,'output_tokens':5,'latency_seconds':1.,'cost_usd':0.}}

class FinalAnalysisTests(unittest.TestCase):
    def test_abstentions_penalize_positive_recall_and_all_case_accuracy(self):
        rows=[record('a',True,True),record('b',False,False),record('c',None,True),record('d',None,False)]
        m=independent_metrics(rows,{})
        self.assertEqual((m['weak_tp'],m['weak_fp'],m['weak_fn'],m['weak_tn']),(1,0,1,1))
        self.assertEqual(m['weak_precision'],1)
        self.assertEqual(m['weak_recall'],.5)
        self.assertAlmostEqual(m['weak_f1'],2/3)
        self.assertEqual(m['weak_flag_accuracy'],.5)
        self.assertEqual(m['flag_abstention_rate'],.5)

    def test_undefined_precision_is_not_invented_zero(self):
        m=independent_metrics([record('a',False,False)],{})
        self.assertIsNone(m['weak_precision']); self.assertIsNone(m['weak_recall']); self.assertIsNone(m['weak_f1'])

    def test_paired_bootstrap_joins_ids_and_ignores_unmatched_cases(self):
        a=[record('b',False,False),record('a',True,True),record('left',False,True)]
        b=[record('a',True,True),record('right',True,False),record('b',False,False)]
        m=bootstrap_difference(a,b,iterations=100)
        self.assertEqual(m['paired_n'],2)
        self.assertEqual(m['delta_f1'],0)
        self.assertEqual((m['ci_low'],m['ci_high']),(0,0))

    def test_category_abstention_counts_as_disagreement(self):
        m=independent_metrics([record('a',None,False,None)],{})
        self.assertEqual(m['current_agreement'],0)
        self.assertEqual(m['original_agreement'],0)
        self.assertEqual(m['category_abstention_rate'],1)

if __name__=='__main__': unittest.main()
