from linalg.test_linalg import (scopy_verify,
                                saxpy_verify,
                                sgemv_verify,
                                sgemm_verify)

def test_scopy():
   scopy_verify()

def test_saxpy():
    saxpy_verify()

def test_sgemv():
    sgemv_verify()

def test_sgemm():
    sgemm_verify()
