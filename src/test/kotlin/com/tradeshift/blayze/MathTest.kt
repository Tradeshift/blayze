package com.tradeshift.blayze

import org.junit.Test

import org.junit.Assert.*

class MathTest {
    @Test
    fun test_logStudentT() {
        //values from scipy.stats.t.logpdf
        assertEquals(-2.22524089938426, logStudentT(2.1, 3.0, 2.2, 3.4), 1e-6)
        assertEquals(-3.5334054763377027, logStudentT(12.21, 13.0, 12.2, 13.4), 1e-6)
    }

    @Test
    fun test_logBeta(){
        //values from scipy.special.betaln
        assertEquals(-6.20455776256869, logBeta(9.0, 3.0), 1e-6)
        assertEquals(-34.456187920349265, logBeta(19.0, 33.0), 1e-6)
    }
}