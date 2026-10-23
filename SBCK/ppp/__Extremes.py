
## Copyright(c) 2023 / 2026 Yoann Robin
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

###############
## Libraries ##
###############

import warnings
import numpy as np
from .__PrePostProcessing import PrePostProcessingPerCols
from ..misc.__sys import deprecated


############
## Typing ##
############

from typing import Any

_Array = np.ndarray


###########
## Class ##
###########

class LimitTailsRatio(PrePostProcessingPerCols):##{{{
    """This class is used to post-process the tails of the correction, in the
    case where too large values are produced.
    
    The idea (for the right tail) is to compare the max of the correction with
    the bound
        B = Q50(Y0) + S * (max(Y0) - Q50(Y0)) + r * (Q50(X1) - Q50(X0)),
    where S = min( ratio , (Q95 - Q50)(X1) / (Q95 - Q50)(X0) ), and r = 1, or
    std(Y0) / std(X0) if norm = "dynamical". If the max is greater than B, the
    tail beyond the quantile 95% is rescaled to the interval [Q95%, B]. The
    left tail works the same way, with Q5 and min(Y0).
    
    If the max is greater than B but B is not beyond the quantile 95% of the
    correction (for example when values are tied at the max), or B is NaN, the
    tail cannot be rescaled to [Q95%, B] and is left unchanged. These tails are
    flagged in `flags` (after predict; one value per column):
    'LimitTailsRatio.bound_inside_right' and 'LimitTailsRatio.bound_inside_left'.
    
    For precipitation, use tails = "right", with SBCK.ppp.SSR in the pipe of
    this class. NaN values of the projection period are ignored by the
    quantiles and the max (min), and stay NaN.
    
    """
    
    ## Attributes ##{{{
    flag_names: tuple[str,...] = ( "LimitTailsRatio.bound_inside_right" , "LimitTailsRatio.bound_inside_left" )
    flags: dict[str,_Array]
    _ratio: float
    _tails: str
    _norm: str
    
    _p_r: float
    _p_l: float
    _p_c: float
    
    _qcY0:  float | _Array = np.nan
    _maxY0: float | _Array = np.nan
    _minY0: float | _Array = np.nan
    _qrX0:  float | _Array = np.nan
    _qcX0:  float | _Array = np.nan
    _qlX0:  float | _Array = np.nan
    _qrX1:  float | _Array = np.nan
    _qcX1:  float | _Array = np.nan
    _qlX1:  float | _Array = np.nan
    _stdY0: float | _Array = np.nan
    _stdX0: float | _Array = np.nan
    _stdX1: float | _Array = np.nan
    
    ##}}}

    ## __init__( self, ... ): ##{{{
    def __init__( self, *args: Any, ratio: float = 1.5,
                 p_r: float = 0.95, p_l: float = 0.05, p_c: float = 0.5,
                 tails: str = "both", norm: str = "origin",
                 **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        ratio: float
            Maximal ratio to increase observed extreme
        p_r: float
            Right quantile to estimated the right tail. 95% in the example.
        p_l: float
            Left quantile to estimate the left tails. 5% in the example.
        p_c: float
            Center quantile to estimated the tails. 50% in the example.
        tails: str
            Tails to apply the PPP. Can be "left", "right" or "both".
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        super().__init__( *args, **kwargs )
        self._name = "LimitTailsRatio"
        
        self._ratio = ratio
        self._tails = tails
        self._norm  = norm
        
        self._p_r   = p_r
        self._p_l   = p_l
        self._p_c   = p_c
        
        self.flags  = {}
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """transform"""
        
        if self._kind == "Y0":
            self._qcY0  = np.quantile( X , self._p_c , axis = 0 )
            self._maxY0 = np.max( X , axis = 0 )
            self._minY0 = np.min( X , axis = 0 )
            self._stdY0 = np.std( X , axis = 0 )
        if self._kind == "X0":
            self._qrX0 = np.quantile( X , self._p_r , axis = 0 )
            self._qcX0 = np.quantile( X , self._p_c , axis = 0 )
            self._qlX0 = np.quantile( X , self._p_l , axis = 0 )
            self._stdX0 = np.std( X , axis = 0 )
        if self._kind == "X1":
            ## NaN values to correct are ignored (a column of NaN only gives NaN)
            with warnings.catch_warnings():
                warnings.simplefilter( "ignore" , RuntimeWarning )
                self._qrX1 = np.nanquantile( X , self._p_r , axis = 0 )
                self._qcX1 = np.nanquantile( X , self._p_c , axis = 0 )
                self._qlX1 = np.nanquantile( X , self._p_l , axis = 0 )
                self._stdX1 = np.nanstd( X , axis = 0 )
        
        return X
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """inverse transform"""
        X  = Xt.copy()
        if self._kind == "X1":
            
            ratio = 1
            if self._norm == "dynamical":
                ratio = self._stdY0 / self._stdX0
            
            cols   = np.zeros( Xt.shape[1] , dtype = bool )
            cols[self.cols] = True
            inside = { "right" : np.zeros( Xt.shape[1] , dtype = bool ) , "left" : np.zeros( Xt.shape[1] , dtype = bool ) }

            ## Right tail
            if self._tails in ["right","both"]:
                SR  = (self._qrX1 - self._qcX1) / (self._qrX0 - self._qcX0)
                SR  = np.where( SR > self._ratio , self._ratio , SR ) * ( self._maxY0 - self._qcY0 ) + self._qcY0 + (self._qcX1 - self._qcX0) * ratio
                with warnings.catch_warnings():
                    warnings.simplefilter( "ignore" , RuntimeWarning )
                    MR = np.nanmax( Xt , axis = 0 )
                    QR = np.nanquantile( Xt , self._p_r , axis = 0 )
                inside["right"] = ~(MR < SR) & ~(QR < SR) & ~np.isnan(MR) & cols
                with np.errstate( divide = "ignore" , invalid = "ignore" ):
                    Xt = np.where( (MR < SR) | ~(QR < SR) | (Xt < QR) , Xt , (Xt - QR) / (MR - QR) * (SR - QR) + QR )
                X[:,self.cols] = Xt[:,self.cols]
            
            ## Left tail
            if self._tails in ["left","both"]:
                SL  = (self._qcX1 - self._qlX1) / (self._qcX0 - self._qlX0)
                SL  = np.where( SL > self._ratio , self._ratio , SL ) * ( self._minY0 - self._qcY0) + self._qcY0 + (self._qcX1 - self._qcX0) * ratio
                with warnings.catch_warnings():
                    warnings.simplefilter( "ignore" , RuntimeWarning )
                    ML = np.nanmin( Xt , axis = 0 )
                    QL = np.nanquantile( Xt , self._p_l , axis = 0 )
                inside["left"] = ~(ML > SL) & ~(QL > SL) & ~np.isnan(ML) & cols
                with np.errstate( divide = "ignore" , invalid = "ignore" ):
                    Xt = np.where( (ML > SL) | ~(QL > SL) | (Xt > QL) , Xt , (Xt - QL) / (ML - QL) * (SL - QL) + QL )
                X[:,self.cols] = Xt[:,self.cols]
            
            self.flags = { n : inside[s] for n,s in zip(self.flag_names,("right","left")) }
            
        return X
    ##}}}
    
##}}}


################
## Deprecated ##
################

@deprecated( "PPPLimitTailsRatio is renamed LimitTailsRatio since the version 2.0.0" )
class PPPLimitTailsRatio(LimitTailsRatio):##{{{
    
    """See SBCK.ppp.LimitTailsRatio"""

    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPLimitTailsRatio"
    ##}}}
    
##}}}

