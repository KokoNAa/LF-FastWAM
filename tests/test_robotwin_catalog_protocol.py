import json
import pytest
from experiments.robotwin.catalog_protocol import validate_namespace, excluded_scene_records


def test_catalog_namespaces_are_disjoint_and_check_the_entire_attempt_window():
    validate_namespace('dev', 91300000, 25)
    validate_namespace('test', 91700000, 100)
    for split, seed, attempts in [('test',91300000,25),('dev',91700000,25),
                                  ('test',91799990,11),('test',91700000,0)]:
        with pytest.raises(ValueError):
            validate_namespace(split, seed, attempts)


def test_exclusions_bind_both_training_manifests_and_catalog_records(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'states':[{'scene_seed':80000000}]}))
    catalog=tmp_path/'episodes.jsonl';catalog.write_text(json.dumps({'scene_seed':91700003})+'\n')
    seeds, receipts=excluded_scene_records([manifest,catalog],split='test')
    assert seeds=={80000000,91700003}
    assert len(receipts)==2 and all(len(r['sha256'])==64 and r['records']==1 for r in receipts)
    catalog.write_text(json.dumps({'scene_seed':91700004})+'\n')
    changed, updated=excluded_scene_records([manifest,catalog],split='test')
    assert 91700003 not in changed and updated[1]['sha256']!=receipts[1]['sha256']


def test_test_split_cannot_silently_omit_or_accept_invalid_exclusions(tmp_path):
    with pytest.raises(ValueError,match='require explicit'):
        excluded_scene_records([],split='test')
    path=tmp_path/'manifest.json'
    for content in [{'states':[]},{'states':[{}]},{'states':[{'scene_seed':True}]}]:
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError):
            excluded_scene_records([path],split='test')
    assert excluded_scene_records([],split='dev')== (set(),[])
