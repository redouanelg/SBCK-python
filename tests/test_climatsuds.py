#!/usr/bin/env python3 -m unittest

## Copyright(c) 2026 Redouane Lguensat
##
## This file is part of SBCK.
##
## SBCK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## SBCK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with SBCK.  If not, see <https://www.gnu.org/licenses/>.

"""Regression tests of the climatsuds branch: one test (or more) per fix that
is not in upstream SBCK. Run in the tests directory with
    python -m unittest test_climatsuds
"""

#############
## Imports ##
#############

import sys
import subprocess
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sc
import dask

import SBCK as bc
import SBCK.ppp as bcp
import SBCK.stats as bcs
import SBCK.clim as bcc
from SBCK.clim.__tools import yearly_window

try:
    import zxarray as zr
    import distributed
except Exception:
    zr = None


#############
## Helpers ##
#############

def _pr( rng , n , p0 , k , s ):
    """Precipitation-like sample: a fraction p0 of zeros, gamma(k,s) elsewhere"""
    return np.where( rng.random(n) < p0 , 0.0 , rng.gamma( k , s , n ) )

class _Recorder:
    """Bias correction stand-in: records the years of the fit and predict windows, returns X1 unchanged"""
    windows = []
    def fit( self , Y0 , X0 , X1 ):
        self._fit = (int(X1.min()),int(X1.max()))
        return self
    def predict( self , X1 ):
        _Recorder.windows.append( self._fit + (int(X1.min()),int(X1.max())) )
        return X1

class _Failing(bc.CDFt):
    """CDFt whose fit always fails"""
    def fit( self , *args , **kwargs ):
        raise RuntimeError("fit failed")


###########
## Tests ##
###########

class Test_ClimFlags(unittest.TestCase):##{{{
    """SBCK.clim collects the diagnostic flags of the methods"""

    def setUp(self):
        rng  = np.random.default_rng(5)
        t    = pd.date_range( "2001-01-01" , "2010-12-31" , freq = "D" )
        y    = rng.normal( 0 , 1 , (t.size,4) )
        x    = rng.normal( 0.5 , 1.2 , (t.size,4) )
        y[:,:2] = np.minimum( y[:,:2] , np.quantile( y[:,:2] , 0.93 , axis = 0 ) )
        late = t.year >= 2006
        x[late,3] = 0.9 + 0.3 * rng.normal( size = late.sum() )
        c = { "time" : t , "cell" : [0,10,20,30] }
        self.Y,self.X = xr.DataArray( y , dims = ("time","cell") , coords = c ),xr.DataArray( x , dims = ("time","cell") , coords = c )
        self.kw = dict( seas_cycle = "season" , chunks = { "cell" : 2 } , calibration_period = (2001,2005) ,
                        projection_range = (2006,2010) , projection_window = (1,3,1) )

    def test_table_equals_direct_recomputation(self):
        ltr = dict( bc_method = bcp.LimitTailsRatio , bc_method_kwargs = dict( bc_method = bc.CDFt ) )
        Z,F = bcc.apply_bcm_along_time( self.Y , self.X , return_flags = True , **ltr , **self.kw )
        Zn  = bcc.apply_bcm_along_time( self.Y , self.X , **ltr , **self.kw )
        np.testing.assert_array_equal( Z.values , Zn.values )
        yr,se = self.X.time.dt.year.values,self.X.time.dt.season.values
        y,x   = self.Y.values,self.X.values
        want  = set()
        for j,cell in enumerate(self.X.cell.values):
            for g,s in enumerate( ["MAM","JJA","SON","DJF"] , start = 1 ):
                m = se == s
                for tf0,tp0,tp1,tf1 in [ tuple(int(v) for v in w) for w in yearly_window( 2006 , 2010 , 1 , 3 , 1 , 2001 , 2010 ) ]:
                    o = bcp.LimitTailsRatio( bc_method = bc.CDFt ).fit( y[m & (yr <= 2005),j][:,None] , x[m & (yr <= 2005),j][:,None] ,
                                                                       x[m & (yr >= tf0) & (yr <= tf1),j][:,None] )
                    o.predict( x[m & (yr >= tp0) & (yr <= tp1),j][:,None] )
                    want |= { (int(cell),g,f"{tp0}-{tp1}",n) for n,v in o.flags.items() if v[0] }
        got = { (int(r.cell),int(r.group),r.window,r.flag) for _,r in F.iterrows() }
        self.assertGreater( len(want) , 0 )
        self.assertEqual( got , want )

    def test_method_without_flags(self):
        Z,F = bcc.apply_bcm_along_time( self.Y , self.X , bc_method = bc.CDFt , return_flags = True , **self.kw )
        self.assertIsNone(F)
        np.testing.assert_array_equal( Z.values , bcc.apply_bcm_along_time( self.Y , self.X , bc_method = bc.CDFt , **self.kw ).values )
##}}}

class Test_LimitTailsRatio(unittest.TestCase):##{{{
    """A tail whose bound is inside it is left unchanged and flagged"""

    def test_bound_inside_tail(self):
        rng   = np.random.default_rng(1)
        Y0,X0 = rng.normal( 300 , 1 , 360 ),rng.normal( 300 , 1 , 360 )
        X1    = np.r_[rng.normal( 300 , 3 , 300 ),rng.normal( 285 , 2 , 60 )]
        Z1    = bc.CDFt( norm = "None" ).fit( Y0 , X0 , X1 ).predict(X1).ravel()
        ltr   = bcp.LimitTailsRatio( ratio = 1 , tails = "left" , bc_method = bc.CDFt , bc_method_kwargs = { "norm" : "None" } )
        Z     = ltr.fit( Y0 , X0 , X1 ).predict(X1).ravel()
        self.assertTrue( np.isfinite(Z).all() )
        self.assertEqual( Z[Z1.argmin()] , Z.min() )
        self.assertTrue( ltr.flags["LimitTailsRatio.bound_inside_left"][0] )
        self.assertFalse( ltr.flags["LimitTailsRatio.bound_inside_right"][0] )
##}}}

class Test_QM(unittest.TestCase):##{{{
    """QM outside the calibration range, and with NaN"""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.Y0,self.X0 = self.rng.normal( size = 500 ),self.rng.normal( size = 500 )
        self.qm = bc.QM().fit( self.Y0 , self.X0 )

    def test_all_outside(self):
        qm,X0 = self.qm,self.X0
        top   = np.asarray( qm.predict( np.r_[X0.max() + 1,0.0] ) ).ravel()[0]
        self.assertFalse( qm.flags["QM.outside_calibration"][0] )
        Z = np.asarray( qm.predict( X0.max() + 1 + self.rng.random(300) ) ).ravel()
        self.assertTrue( np.all( Z == top ) )
        self.assertTrue( qm.flags["QM.outside_calibration"][0] )
        bottom = np.asarray( qm.predict( np.r_[X0.min() - 1,0.0] ) ).ravel()[0]
        Z = np.asarray( qm.predict( X0.min() - 1 - self.rng.random(50) ) ).ravel()
        self.assertTrue( np.all( Z == bottom ) )
        self.assertTrue( qm.flags["QM.outside_calibration"][0] )

    def test_nan_kept(self):
        X1  = self.rng.normal( size = (200,2) )
        X1n = X1.copy()
        X1n[[3,50,120],0] = np.nan
        Pp  = np.abs(self.rng.normal( size = (500,2) )) * (self.rng.random((500,2)) > 0.3)
        for kw,Y0,X0 in [ ({},self.Y0,self.X0) , (dict( rvY0 = bcs.rv_empirical_ratio , rvX0 = bcs.rv_empirical_ratio ),Pp[:,0],Pp[:,1]) ]:
            qm = bc.QM(**kw).fit( np.column_stack([Y0,Y0]) , np.column_stack([X0,X0]) )
            Zc = np.asarray( qm.predict( np.abs(X1) if kw else X1 ) )
            Zn = np.asarray( qm.predict( np.where( np.isnan(X1n) , np.nan , np.abs(X1) if kw else X1 ) ) )
            np.testing.assert_array_equal( np.isnan(Zn) , np.isnan(X1n) )
            np.testing.assert_array_equal( Zn[~np.isnan(X1n)] , Zc[~np.isnan(X1n)] )
        qm.predict( np.column_stack( [np.full(10,np.nan),np.full(10,1e9)] ) )
        np.testing.assert_array_equal( qm.flags["QM.outside_calibration"] , [False,True] )
##}}}

class Test_NaN_kept(unittest.TestCase):##{{{
    """NaN values stay NaN in SSR, rv_empirical_ratio, LFLogistic and PreserveOrder"""

    def test_SSR(self):
        rng   = np.random.default_rng(3)
        Y0,X0,X1 = ( np.column_stack( [_pr( rng , 400 , 0.4 , 0.8 , 4 ) for _ in range(2)] ) for _ in range(3) )
        X1n   = X1.copy()
        X1n[[5,77,301],0] = np.nan
        for method in [bc.QM,bc.CDFt]:
            np.random.seed(5)
            ssr = bcp.SSR( bc_method = method ).fit( Y0 , X0 ) if method is bc.QM else bcp.SSR( bc_method = method ).fit( Y0 , X0 , X1 )
            np.random.seed(6); Zc = np.asarray( ssr.predict(X1) )
            np.random.seed(6); Zn = np.asarray( ssr.predict(X1n) )
            np.testing.assert_array_equal( np.isnan(Zn) , np.isnan(X1n) )
            np.testing.assert_array_equal( Zn[~np.isnan(X1n)] , Zc[~np.isnan(X1n)] )
        Xt = np.array( [[np.nan,1.0],[ssr.Xn[0,0] / 2,np.nan],[5.0,5.0]] )
        X  = ssr.itransform(Xt)
        np.testing.assert_array_equal( np.isnan(X) , np.isnan(Xt) )
        self.assertEqual( X[1,0] , 0 )

    def test_rv_empirical_ratio(self):
        rng = np.random.default_rng(4)
        rv  = bcs.rv_empirical_ratio.fit( _pr( rng , 500 , 0.3 , 0.7 , 5 ) )
        p   = rv.cdf( np.array([np.nan,0.0,1.0]) )
        x   = rv.icdf( np.array([np.nan,rv.p0,0.9]) )
        self.assertTrue( np.isnan(p[0]) and np.isnan(x[0]) )
        self.assertEqual( p[1] , rv.p0 )
        self.assertEqual( x[1] , 0 )
        self.assertTrue( np.isfinite(p[1:]).all() and np.isfinite(x[1:]).all() )

    def test_LFLogistic(self):
        lf = bcp.LFLogistic( ymin = 0.0 , ymax = 1.0 )
        y  = np.asarray( lf.transform( np.array([[np.nan],[0.3]]) ) ).ravel()
        x  = np.asarray( lf.itransform( np.array([[np.nan],[0.0]]) ) ).ravel()
        self.assertTrue( np.isnan(y[0]) and np.isfinite(y[1]) )
        self.assertTrue( np.isnan(x[0]) )
        self.assertAlmostEqual( x[1] , 0.5 )

    def test_PreserveOrder(self):
        po = bcp.PreserveOrder( cols_order = [0,1,2] )
        X  = np.array( [[3.0,1.0,2.0],[3.0,np.nan,1.0],[np.nan,np.nan,np.nan]] )
        with warnings.catch_warnings( record = True ) as w:
            warnings.simplefilter("always")
            Z = np.asarray( po.itransform(X) )
        self.assertEqual( len(w) , 0 )
        np.testing.assert_array_equal( Z[0] , [1.0,2.0,3.0] )
        np.testing.assert_array_equal( Z[1] , [1.0,np.nan,3.0] )
        self.assertTrue( np.isnan(Z[2]).all() )
##}}}

class Test_dOTC1d(unittest.TestCase):##{{{
    """dOTC1d: clear error with a constant X0, reinfer_X1 / reinfer_X0"""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.Y0,self.X0,self.X1 = rng.normal( 0 , 1 , (400,2) ),rng.normal( 1 , 2 , (400,2) ),rng.normal( 2 , 2 , (400,2) )

    def test_constant_X0(self):
        X0 = self.X0.copy()
        X0[:,0] = 1.5
        with self.assertRaisesRegex( ValueError , "X0 is constant" ):
            bc.dOTC1d().fit( self.Y0 , X0 , self.X1 )

    def test_reinfer(self):
        o = bc.dOTC1d().fit( self.Y0 , self.X0 , self.X1 )
        Z1,Z0 = o.predict( self.X1 , self.X0 )
        for kw in [ { "reinfer_X1" : True } , { "reinfer_X0" : True } ]:
            R1,R0 = o.predict( self.X1 , self.X0 , **kw )
            np.testing.assert_array_equal( R1 , Z1 )
            np.testing.assert_array_equal( R0 , Z0 )
##}}}

class Test_CDFt_laws(unittest.TestCase):##{{{
    """CDFt with unbounded laws, and rv_empirical_gpd with ties at a tail"""

    def test_unbounded_laws(self):
        rng = np.random.default_rng(0)
        Y0,X0,X1 = rng.normal( 0 , 1 , 360 ),rng.normal( 1 , 2 , 360 ),rng.normal( 2 , 2.5 , 300 )
        kw  = dict( rvY0 = sc.norm , rvX0 = sc.norm , rvX1 = sc.norm )
        for oob in ["Y0","Y0CC","CC"]:
            with self.assertRaisesRegex( ValueError , "unbounded support" ):
                bc.CDFt( oob = oob , **kw ).fit( Y0 , X0 , X1 ).predict(X1)
        Z = bc.CDFt( oob = "None" , **kw ).fit( Y0 , X0 , X1 ).predict(X1)
        self.assertTrue( np.isfinite(Z).all() )
        Z0 = bc.CDFt( oob = "Y0" , **kw ).fit( Y0 , X0 , X1 ).ubcm[0]._predictZ0(X0)
        self.assertTrue( np.isfinite(Z0).all() )

    def test_gpd_tail_with_ties(self):
        rng = np.random.default_rng(0)
        rv  = bcs.rv_empirical_gpd.fit( _pr( rng , 360 , 0.5 , 0.8 , 6 ) )
        self.assertTrue( np.isfinite( rv.cdf( np.linspace( -1 , 60 , 200 ) ) ).all() )
        self.assertTrue( np.isfinite( rv.icdf( np.linspace( 0.001 , 0.999 , 200 ) ) ).all() )
        for p in ["scalel","shapel","scaler","shaper"]:
            self.assertTrue( np.isfinite( getattr( rv , p ) ) )
##}}}

class Test_Refit(unittest.TestCase):##{{{
    """A second fit gives the same result as a new object"""

    def test_refit(self):
        rng = np.random.default_rng(2)
        Y0,X0,X1 = rng.normal( 0 , 1 , (300,2) ),rng.normal( 1 , 2 , (300,2) ),rng.normal( 2 , 2 , (300,2) )
        qm = bc.QM().fit( Y0 , X0 ).fit( Y0 + 100 , X0 )
        np.testing.assert_array_equal( qm.predict(X1) , bc.QM().fit( Y0 + 100 , X0 ).predict(X1) )
        cdft = bc.CDFt().fit( Y0 , X0 , X1 ).fit( Y0[:,:1] , X0[:,:1] , X1[:,:1] )
        np.testing.assert_array_equal( cdft.predict(X1[:,:1]) , bc.CDFt().fit( Y0[:,:1] , X0[:,:1] , X1[:,:1] ).predict(X1[:,:1]) )
##}}}

class Test_ClimWindows(unittest.TestCase):##{{{
    """SBCK.clim: seas_cycle = "window" keeps the projection window, dask-backed inputs"""

    def test_projection_window(self):
        t    = pd.date_range( "1951-01-01" , "2010-12-31" , freq = "D" )
        year = xr.DataArray( np.repeat( t.year.values[:,None] , 2 , axis = 1 ).astype(float) , dims = ("time","lat") ,
                             coords = { "time" : t , "lat" : [0.0,1.0] } )
        _Recorder.windows.clear()
        with dask.config.set( scheduler = "synchronous" ):
            bcc.apply_bcm_along_time( year , year , bc_method = _Recorder , calibration_period = (1951,1980) , projection_range = (1951,2010) ,
                                      projection_window = (5,10,5) , seas_cycle = "window" , seas_cycle_window = (15,30,15) )
        self.assertEqual( { b - a + 1 for a,b,_,_ in _Recorder.windows } , {20} )

    def test_dask_inputs(self):
        rng = np.random.default_rng(6)
        t   = pd.date_range( "2001-01-01" , "2010-12-31" , freq = "D" )
        c   = { "time" : t , "cell" : [0,1,2] }
        Y,X = ( xr.DataArray( rng.normal( m , 1 , (t.size,3) ) , dims = ("time","cell") , coords = c ) for m in (0,1) )
        kw  = dict( bc_method = bc.QDM , calibration_period = (2001,2005) , projection_window = (1,3,1) )
        Z   = bcc.apply_bcm_along_time( Y , X , **kw )
        Zd  = bcc.apply_bcm_along_time( Y.chunk( { "time" : 365 } ) , X.chunk( { "time" : 365 } ) , **kw )
        np.testing.assert_array_equal( np.asarray(Zd.values) , Z.values )

    @unittest.skipIf( zr is None , "zxarray is not installed" )
    def test_zapply_cluster_closed_on_error(self):
        rng = np.random.default_rng(7)
        t   = pd.date_range( "1961-01-01" , "1966-12-31" , freq = "D" )
        Y   = xr.DataArray( rng.normal( size = (t.size,2) ) , dims = ("time","lat") , coords = { "time" : t , "lat" : [0.0,1.0] } )
        zY  = zr.ZXArray.from_xarray(Y)
        with self.assertRaises(Exception):
            bcc.zapply_bcm_along_time( zY , zY , bc_method = _Failing , calibration_period = (1961,1963) , projection_window = (1,1,1) )
        with self.assertRaises(ValueError):
            distributed.default_client()
##}}}

class Test_Ranks(unittest.TestCase):##{{{
    """Rank shuffles with SciPy >= 1.18 (rankdata returns float ranks)"""

    def test_schaake_shuffle(self):
        rng   = np.random.default_rng(7)
        Y0,X1 = rng.normal( size = (300,3) ),rng.normal( 2 , 1 , (300,3) )
        Z     = bcs.SchaakeShuffleRef( 0 , Y0 ).predict(X1)
        np.testing.assert_array_equal( np.sort( Z , axis = 0 ) , np.sort( X1 , axis = 0 ) )
##}}}


##########
## main ##
##########

if __name__ == "__main__":
    unittest.main( argv = sys.argv[:1] )
