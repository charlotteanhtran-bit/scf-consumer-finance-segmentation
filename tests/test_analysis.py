import unittest
import numpy as np
import pandas as pd
from analyze import prepare_sample, transform_features, FINANCIAL_FEATURES

class PreparationTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for family in range(25):
            for imp in range(1,6):
                rows.append(dict(YY1=family,Y1=10*family+imp,WGT=1,TURNFEAR=1,**{c:100 for c in FINANCIAL_FEATURES}))
        return pd.DataFrame(rows)
    def test_one_record_per_household(self):
        for imp in range(1,6):
            result=prepare_sample(self.frame(),imp)
            self.assertEqual(len(result),25)
            self.assertTrue(result.YY1.is_unique)
            self.assertTrue(result.IMPLIC.eq(imp).all())
    def test_duplicate_family_rejected(self):
        df=self.frame()
        with self.assertRaises(ValueError): prepare_sample(pd.concat([df,df.iloc[[0]]]))
    def test_signed_transform_preserves_negative_and_zero(self):
        out=transform_features(pd.DataFrame({'x':[-100,0,100]}),['x'])
        np.testing.assert_allclose(out.x,[-np.log1p(100),0,np.log1p(100)])
