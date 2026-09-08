#!/usr/bin/env python3
"""Summarize a COMPLETE fixed600 matrix without changing frozen scoring rules."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

TASK_NAMES = dict(blocks_ranking_rgb='排序 RGB→BGR', stack_blocks_two='叠块：红在绿上',
                  place_a2b_left='左放置→右放置',place_a2b_right='右放置→左放置',
                  place_burger_fries='汉堡/薯条交换槽位')
ARM_NAMES = dict(no_eraf='no-eraf',eraf_only='ERAF',eraf_fg='ERAF+FG')


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ratio(n,d):return n/d if d else None


def summarize_cell(episodes):
    if not episodes:raise ValueError('Empty cell')
    metrics=[e['manipulation_metrics'] for e in episodes]
    n=len(episodes)
    cf=sum(e['counterfactual_goal_ever_success'] for e in episodes)
    lift=sum(m['any_correct_object_lifted'] for m in metrics)
    after=sum(m['full_goal_after_any_lift'] for m in metrics)
    strict=sum(m['correct_placement_after_lift'] for m in metrics)
    for e,m in zip(episodes,metrics):
        if m['full_goal_after_any_lift'] and not (e['counterfactual_goal_ever_success'] and m['any_correct_object_lifted']):
            raise ValueError('Invalid task-success-after-lift attribution')
        if m['correct_placement_after_lift'] and not m['full_goal_after_any_lift']:
            raise ValueError('Strict placement must also satisfy the original goal')
        if m['physics_samples']<1:raise ValueError('Missing physical samples')
    return dict(episodes=n,cf=cf,cf_rate=cf/n,lift=lift,lift_rate=lift/n,
                cf_after_lift=after,task_placement_given_lift_rate=ratio(after,lift),
                strict_release_placement=strict,strict_release_given_lift_rate=ratio(strict,lift),
                cf_without_verified_lift=sum(e['counterfactual_goal_ever_success'] and not m['any_correct_object_lifted'] for e,m in zip(episodes,metrics)),
                lift_without_later_task_success=lift-after,
                all_instruction_objects_lifted=sum(m['all_instruction_objects_lifted'] for m in metrics),
                final_strict_slots=sum(all(m['final_strict_slot_assignment'].values()) for m in metrics if m['final_strict_slot_assignment'] is not None))


def report(root):
    root=Path(root)
    plan=read(root/'frozen_protocol.json');status=read(root/'status.json');terminal=read(root/'terminal_verification.json')
    if not (status['complete'] and status['status']=='complete' and terminal['complete'] and terminal['episodes']==600):
        raise ValueError('The complete600 evaluation and terminal verification are required')
    if set(plan['tasks'])!=set(TASK_NAMES) or set(plan['models'])!=set(ARM_NAMES):
        raise ValueError('Formal task/model scope changed')
    cells=[];all_rows={};bindings={};objects=[]
    for arm,model in plan['models'].items():
        all_rows[arm]=[]
        for task in plan['tasks']:
            base=root/'evaluation'/arm/task/'demo_clean/counterfactual'
            p=base/'episodes.jsonl';episodes=rows(p);bindings[str(p.relative_to(root))]=sha(p)
            canonical=root/'catalog'/task/'demo_clean/correct/episodes.jsonl';expected=rows(canonical)
            complete=read(base/'complete.json');initial=read(base/'initial_states.json')
            if len(episodes)!=40 or len(expected)!=40 or len(initial)!=40 or not complete['complete']:
                raise ValueError('Formal40 cell incomplete')
            if complete['checkpoint_sha256']!=model['sha256'] or complete['canonical_sha256']!=sha(canonical):
                raise ValueError('Checkpoint/catalog binding changed')
            if (complete['eraf']!=model['eraf'] or complete['policy_kind']!='repair' or complete['memory_mode']!='carry'
                or complete['deployment']!=dict(action_horizon=32,replan_steps=24,inference_steps=10) or not complete['manipulation_metrics']):
                raise ValueError('Inference protocol changed')
            for e,c,h in zip(episodes,expected,initial):
                for key in ['episode_index','scene_seed','source_instruction','counterfactual_instruction']:
                    if e[key]!=c[key]:raise ValueError('Scene/instruction mismatch')
                if e['condition']!='counterfactual' or e['instruction_goal']!='counterfactual' or e['selected_goal']!='counterfactual':
                    raise ValueError('Wrong condition')
                if h!=dict(scene_seed=c['scene_seed'],sha256=c['initial_physical_state_sha256']):
                    raise ValueError('Physical initial state mismatch')
            cells.append(dict(arm=arm,task=task,**summarize_cell(episodes)))
            all_rows[arm].extend(episodes)
            object_names=set(episodes[0]['manipulation_metrics']['objects'])
            for obj in sorted(object_names):
                values=[e['manipulation_metrics']['objects'][obj] for e in episodes]
                lifted=sum(v['correctly_lifted'] for v in values)
                placed=sum(v['placed_after_lift'] for v in values)
                objects.append(dict(arm=arm,task=task,object=obj,episodes=40,lifted=lifted,
                                    strict_placement_after_lift=placed,strict_final_placement=sum(v['final_placement'] for v in values),
                                    strict_placement_given_lift=ratio(placed,lifted)))
    pooled={arm:summarize_cell(v) for arm,v in all_rows.items()}
    paired=[]
    for a,b in [('no_eraf','eraf_only'),('no_eraf','eraf_fg'),('eraf_only','eraf_fg')]:
        for task in [*plan['tasks'], 'all_five']:
            aa=[e for e in all_rows[a] if task=='all_five' or e['source_task']==task]
            bb=[e for e in all_rows[b] if task=='all_five' or e['source_task']==task]
            if len(aa)!=len(bb):raise ValueError('Unpaired outcomes')
            for x,y in zip(aa,bb):
                if (x['source_task'],x['scene_seed'])!=(y['source_task'],y['scene_seed']):raise ValueError('Unpaired scene ordering')
            gains=sum(not x['counterfactual_goal_ever_success'] and y['counterfactual_goal_ever_success'] for x,y in zip(aa,bb))
            losses=sum(x['counterfactual_goal_ever_success'] and not y['counterfactual_goal_ever_success'] for x,y in zip(aa,bb))
            paired.append(dict(reference=a,candidate=b,task=task,episodes=len(aa),gains=gains,losses=losses,net=gains-losses))
    return dict(format='robotwin_formal_five40_report_v1',complete=True,episodes=600,source_sha256=bindings,
                checkpoint_sha256={a:m['sha256'] for a,m in plan['models'].items()},cells=cells,pooled=pooled,paired=paired,objects=objects,
                definitions=dict(task_placement='Original CF goal reached after any verified correct-object lift.',
                                 strict_release='Additional release/no-contact/height/center/slot checks from the frozen protocol; false is not by itself proof of a placement failure.',
                                 denominator='Task and strict placement conditional rates both divide by episodes with any correct-object lift. Zero denominator is null.',
                                 scope='Five tasks selected before this fresh evaluation; 40 fixed new scenes each; no Correct results or unseen-task claims.'))


def frac(n,d):return f'{n}/{d}（{100*n/d:.1f}%）' if d else '0/0（无可用分母）'


def markdown(r):
    text=['# balanced-target200 五任务正式测评结果','',
          '固定三组200步模型，每任务每模型40个新场景，总计600回合。三组使用相同场景、指令和初始物理状态。','',
          '## CF任务成功','', '| 任务 | no-eraf | ERAF | ERAF+FG |','|---|---:|---:|---:|']
    for task,label in TASK_NAMES.items():
        text.append('| '+label+' | '+' | '.join(frac(next(c for c in r['cells'] if c['arm']==a and c['task']==task)['cf'],40) for a in ARM_NAMES)+' |')
    text+=['| 五任务合计 | '+' | '.join(frac(r['pooled'][a]['cf'],200) for a in ARM_NAMES)+' |','',
           '各任务样本数相同，合计成功率等于五任务宏平均。','',
           '## 夹起与放置','', '| 模型 | 正确夹起/200 | 夹起后达到任务目标/夹起回合 | 严格脱手放置确认/夹起回合 |',
           '|---|---:|---:|---:|']
    for a,label in ARM_NAMES.items():
        x=r['pooled'][a];text.append(f"| {label} | {frac(x['lift'],200)} | {frac(x['cf_after_lift'],x['lift'])} | {frac(x['strict_release_placement'],x['lift'])} |")
    text+=['','“正确夹起”指至少一个指令物体满足双侧真实接触、抬高3厘米、持续0.10秒；全部物体及逐物体结果另存于JSON。',
           '“任务目标”沿用原CF判定。“严格脱手放置”额外要求零夹指接触与位置/高度约束；CF成功时仍可能有轻微接触，因此严格未确认不等于放置失败。两者均未额外执行静置步骤。','',
           '## 同场景配对变化','', '| 比较 | 新增成功 | 丢失成功 | 净变化 |','|---|---:|---:|---:|']
    for p in r['paired']:
        if p['task']=='all_five':text.append(f"| {ARM_NAMES[p['candidate']]} 相对 {ARM_NAMES[p['reference']]} | {p['gains']} | {p['losses']} | {p['net']:+d}/200 |")
    text+=['','## 每任务夹起与放置明细','', '| 任务 | 模型 | 夹起/40 | 夹起后任务目标/夹起 | 严格脱手放置/夹起 |','|---|---|---:|---:|---:|']
    for c in r['cells']:
        text.append(f"| {TASK_NAMES[c['task']]} | {ARM_NAMES[c['arm']]} | {frac(c['lift'],40)} | {frac(c['cf_after_lift'],c['lift'])} | {frac(c['strict_release_placement'],c['lift'])} |")
    text+=['','## 汉堡/薯条严格槽位补充','']
    for a,label in ARM_NAMES.items():
        c=next(c for c in r['cells'] if c['arm']==a and c['task']=='place_burger_fries')
        text.append(f"- {label}：最终更接近指定槽位 {frac(c['final_strict_slots'],40)}。")
    text+=['','本轮只评估用户选定五任务的反事实指令。未测Correct；其余五个任务不进入本轮平均值。结果不能外推为十任务或未见任务表现。','']
    return '\n'.join(text)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();r=report(args.root);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'REPORT.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    (args.output/'REPORT.md').write_text(markdown(r))
    print(json.dumps(dict(complete=True,episodes=r['episodes'],pooled=r['pooled']),ensure_ascii=False))


if __name__=='__main__':main()
