#!/usr/bin/env python
from coffea import hist, processor
import uproot
import sys
import time
import awkward1 as ak
import numpy as np

class MainProcessor(processor.ProcessorABC):
        def __init__(self):
                trigger_axis                = hist.Bin("trigger",               "trigger names",                                    148,    0.0,    148.0 )

                self._accumulator = processor.dict_accumulator({
                        'h_trigger':          hist.Hist("h_trigger",                      trigger_axis),
                })

        @property
        def accumulator(self):
                return self._accumulator

        def process(self, df):
                output = self.accumulator.identity()

                triggerPass = df['TriggerPass']
                triggerNames = df['TriggerNames']

                tPassedList = []
                for evt in triggerPass:
                    tPassed = [tp for tp in range(len(evt)) if evt[tp] == 1]
                    tPassedList.append(tPassed)
                tPassedList = ak.Array(tPassedList)

                oneTrigger = ak.count(tPassedList,axis=-1) > 0

                output['h_trigger'].fill(trigger=ak.flatten(tPassedList),weight=np.ones(len(ak.flatten(tPassedList))))


                return output

        def postprocess(self, accumulator):
                return accumulator


def main():
    tstart = time.time()

    fileset = {
        "2018_QCD_Pt_1000to1400": ["root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV17/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8_0_RA2AnalysisTree.root"]
    }

    # run processor
    output = processor.run_uproot_job(
        fileset,
        treename='TreeMaker2/PreSelection',
        processor_instance=MainProcessor(),
        executor=processor.futures_executor,
        executor_args={'workers': 1, 'flatten': False},
        chunksize=10000,
    )

    fout = uproot.recreate("output.root")
    fout['h_trigger'] = hist.export1d(output['h_trigger'])
    fout.close()

    # print run time in seconds
    dt = time.time() - tstart
    print("run time: %.2f [sec]" % (dt))

if __name__ == "__main__":
    main()
