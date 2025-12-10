#from surd_segments_api import load_surd_from_dir

#  utils file
#load_surd_from_dir("/home/wgh/SOUD/SURD-main/utils")
from DEVCOMSuitepython.utils.surd import surd
from DEVCOMSuitepython.surd_segments import run_surd_segments
import DEVCOMSuitepython.surd_segments as ss
ss.surd = surd
adata = sc.read("/home/wgh/DEVCOM Suite test/tests/testthat/SURD/burkhardt21_merged_flow_learned.h5ad")
res = run_surd_segments(
    adata,
    ctrl_name="Ctrl",#condition control
    treat_name="IFNg",#condition treat
    out_dir     = "./",
    fast_mode   = True
)
#adata = sc.read("/home/wgh/DEVCOM Suite test/tests/testthat/SURD/burkhardt21_merged_flow_learned.h5ad")

# interpreiability
# res = run_surd_segments(
#     adata,
#     ctrl_name="Ctrl",#condition control
#     treat_name="IFNg",#condition treat
#     out_dir     = "/home/wgh/DEVCOM Suite test/tests/testthat/SURD",
#     fast_mode   = True
# )


print(res["in2gem"].shape)
print(res["gem2out"].shape)
print(res["combined"].shape)
print(res["triples"].shape)
