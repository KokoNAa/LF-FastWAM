import pytest
import torch
from scripts.probe_robotwin_deployed_gradients import alignment,gradient_group


def test_alignment_reports_opposing_orthogonal_and_zero_gradients():
    r=alignment({'a':torch.tensor([1.,0.]),'b':torch.tensor([-2.,0.]),'c':torch.tensor([0.,3.]),'d':torch.zeros(2)})
    assert r['pairs']['a|b']=={'dot':-2.,'cosine':-1.}
    assert r['pairs']['a|c']=={'dot':0.,'cosine':0.}
    assert r['pairs']['a|d']['cosine'] is None
    assert r['norms']['b']==2.

@pytest.mark.parametrize('v',[{}, {'a':torch.tensor([float('nan')])}, {'a':torch.ones(2),'b':torch.ones(3)}])
def test_invalid_gradient_vectors_rejected(v):
    with pytest.raises(ValueError):alignment(v)

@pytest.mark.parametrize('row,expected',[
 ({'native_retention':True},'correct_retention'),({'cf_retention':True},'cf_retention'),
 ({'initial_expert_anchor':True,'source_task':'cup'},'initial_cup'),
 ({'fg_correction':True},'fg'),({'ordinary_cf_control':True},'ordinary_cf_control'),({},'ordinary_expert')])
def test_gradient_groups_preserve_ablation_roles(row,expected):
    assert gradient_group(row)==expected


def reports():
    from copy import deepcopy
    records=[dict(step=i//12+1,index=i%12,seed=42+i,pair_id='task',id=str(i),row_sha256=str(i),group='ordinary_expert') for i in range(24)]
    report=dict(complete=True,parameters_unchanged=True,input_hashes_stable=True,optimizer_updates=0,records=records,alignment={})
    a=deepcopy(report);b=deepcopy(report)
    for i in (0,1,2,12,13,14):
        a['records'][i].update(group='ordinary_cf_control')
        b['records'][i].update(group='fg',id='fg'+str(i),row_sha256='fg'+str(i))
    return {'eraf_only':a,'eraf_fg':b}


def test_gradient_comparison_requires_exact_common_rows_and_matched_replacements():
    from scripts.run_robotwin_deployed_gradient_check import compare
    result=compare(reports())
    assert result['matched_common_queries']==18
    assert result['matched_fg_control_positions']==6

@pytest.mark.parametrize('change',['missing','parameters','updates','seed','row','replacement'])
def test_gradient_comparison_rejects_bad_provenance(change):
    from scripts.run_robotwin_deployed_gradient_check import compare
    d=reports();r=d['eraf_fg']
    if change=='missing':r['records'].pop()
    elif change=='parameters':r['parameters_unchanged']=False
    elif change=='updates':r['optimizer_updates']=1
    elif change=='seed':r['records'][0]['seed']=0
    elif change=='row':r['records'][4]['row_sha256']='bad'
    else:r['records'][0]['group']='ordinary_expert'
    with pytest.raises(ValueError):compare(d)
