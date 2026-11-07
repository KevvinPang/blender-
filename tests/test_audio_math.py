import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sam_ellen_livliv import analyze,mouth_weights


class AudioTests(unittest.TestCase):
    def test_silence_has_no_mouth_motion_and_audio_ends_neutral(self):
        silent=analyze(np.zeros(48000))
        self.assertFalse(np.any(silent['open']))
        t=np.arange(48000)/48000
        signal=.2*np.sin(2*np.pi*370*t)*((t>.2)&(t<.7))
        speech=analyze(signal)
        self.assertGreater(speech['open'].max(),.2)
        self.assertEqual(speech['open'][-1],0)
        self.assertTrue(np.isfinite(speech['open']).all())
        self.assertEqual(len(speech['open']),len(speech['times']))

    def test_gain_and_vowel_mix_stay_within_shape_budget(self):
        for opening in np.linspace(0,1,11):
            for vowel in np.linspace(-1,1,11):
                values=mouth_weights(opening,vowel,2)
                self.assertTrue(all(0<=v<=1 for v in values.values()))
                self.assertLessEqual(sum(values.values()),1.00000001)


if __name__=='__main__':unittest.main()
