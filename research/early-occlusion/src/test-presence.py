"""Regression checks for the error budgets and causal visible-fragment protocol."""

from copy import deepcopy
import runpy
import unittest

import numpy as np

from protocol import ROOT, load
from replay import Mode, select


compare = runpy.run_path(str(ROOT/'src/compare-presence.py'))


def observation(boxes, scores):
    return np.asarray(boxes, np.float32).reshape(-1,4), np.asarray(scores,np.float32), np.zeros(len(scores),np.float32)


class ProtocolTests(unittest.TestCase):
    def test_box_budget_counts_multiple_boxes_and_excludes_ties(self):
        frames = [observation([[0,0,10,10],[30,0,40,10]], [.8,.7]),
                  observation([[0,0,10,10]], [.7])]
        threshold = compare['threshold_at_box_budget'](frames, 2)
        self.assertEqual(sum(len(select(o,Mode(threshold))) for o in frames), 1)
        threshold = compare['threshold_at_box_budget'](frames, 0)
        self.assertEqual(sum(len(select(o,Mode(threshold))) for o in frames), 0)
        threshold = compare['threshold_at_budget'](frames, 2)
        self.assertEqual(sum(bool(select(o,Mode(threshold))) for o in frames), 2)
        self.assertEqual(sum(len(select(o,Mode(threshold))) for o in frames), 3)

    def test_confirmation_requires_two_accurate_consecutive_frames(self):
        episode = dict(first=10,last_partial=12,boxes={str(i):[[200,100,210,120]] for i in range(10,13)})
        good = observation([[200,100,210,120]], [.8])
        # Still associated by center distance, but not an accurate visible-body box.
        wrong = observation([[215,100,225,120]], [.8])
        frames = {10:good,11:wrong,12:good}
        result = compare['visible_episode'](episode, frames, .13)
        self.assertTrue(result['strict_first'])
        self.assertIsNone(result['confirmed_frame'])
        frames[11] = good
        result = compare['visible_episode'](episode, frames, .13)
        self.assertEqual(result['confirmed_frame'], 11)
        prefix = dict(episode, last_partial=11)
        self.assertEqual(compare['visible_episode'](prefix,{10:good,11:good},.13), result)

    def test_published_and_candidate_sources_are_disjoint(self):
        annotations = load('next/benchmark.json')
        sources = load('next/sources.json')
        excluded = {r['match_id'] for r in load('sources.json')}
        excluded |= {r['match_id'] for r in load('calibration.json')}
        compare['validate_new'](annotations,sources,excluded)
        bad = deepcopy(sources)
        bad[0]['match_id'] = next(iter(excluded))
        with self.assertRaisesRegex(ValueError,'overlaps'):
            compare['validate_new'](annotations,bad,excluded)

    def test_annotations_require_every_partial_frame_and_no_conflicts(self):
        annotations, sources = load('next/benchmark.json'), load('next/sources.json')
        broken = deepcopy(annotations)
        row = broken['appearances'][0]
        del row['boxes'][str(row['first']+1)]
        with self.assertRaisesRegex(ValueError,'Each partial'):
            compare['validate_new'](broken,sources,[])
        broken = deepcopy(annotations)
        row = broken['appearances'][0]
        broken['empty_windows'].append(dict(case=row['case'],start=row['first'],stop=row['first']))
        with self.assertRaisesRegex(ValueError,'both positive and empty'):
            compare['validate_new'](broken,sources,[])


if __name__ == '__main__':
    unittest.main()
