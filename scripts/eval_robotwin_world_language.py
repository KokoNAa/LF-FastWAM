#!/usr/bin/env python3
"""Run the ordinary CIS loop with only the Video language independently set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]


def instrument(policy, language, seed, journal):
    from experiments.robotwin.world_language_probe import video_language_override
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    if policy.model.policy_guard_enabled or policy.model.uses_transition_queries:
        raise ValueError('Closed-loop primary probe requires no ERAF or transition queries')
    original_begin, original_infer = policy.begin_episode, policy._infer_action_chunk
    state={}

    def begin(env, metadata):
        policy.seed=seed
        state.update(metadata=metadata, calls=0)
        field='source_instruction' if language=='source' else 'counterfactual_instruction'
        state['video_instruction']=metadata[field]
        result=original_begin(env,metadata)
        return result

    def infer(observation, instruction):
        if instruction != state['metadata']['policy_instruction']:
            raise ValueError('Action instruction changed inside an episode')
        with video_language_override(policy.model,DEFAULT_PROMPT.format(task=state['video_instruction'])):
            action=original_infer(observation,instruction)
        if state['calls']==0:
            from experiments.robotwin.no_eraf_probe import observation_hash,CAMERAS
            digest=observation_hash(dict(state=observation['joint_action']['vector'],
                **{c:observation['observation'][c]['rgb'] for c in CAMERAS}))
            journal.write(json.dumps(dict(**state['metadata'],video_language=language,
                video_instruction=state['video_instruction'],action_instruction=instruction,
                noise_seed=seed,initial_observation_sha256=digest))+'\n')
            journal.flush()
        state['calls']+=1
        return action

    policy.begin_episode=begin
    policy._infer_action_chunk=infer
    return policy


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--world-video-language',required=True,choices=['source','target'])
    ap.add_argument('--policy-seed',required=True,type=int)
    args,rest=ap.parse_known_args()
    if '--output' not in rest:raise ValueError('Need output path')
    output=Path(rest[rest.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    import scripts.train_robotwin_cf_decision_adapter as legacy
    import experiments.robotwin.eraf_fg_bridge as repair
    import scripts.eval_robotwin_eraf_fg as evaluate
    legacy_loader,repair_loader=legacy.load_policy,repair.load_policy
    with (output/'world_language_inputs.jsonl').open('x',buffering=1) as journal:
        def wrap(loader):
            def load(*a,**kw):
                policy=loader(*a,**kw)
                policy.model.policy_guard_enabled=False
                return instrument(policy,args.world_video_language,args.policy_seed,journal)
            return load
        legacy.load_policy=wrap(legacy_loader);repair.load_policy=wrap(repair_loader)
        try:
            sys.argv=[str(REPO/'scripts/eval_robotwin_eraf_fg.py'),*rest]
            evaluate.main()
        finally:
            legacy.load_policy=legacy_loader;repair.load_policy=repair_loader
    (output/'world_language_complete.json').write_text(json.dumps(dict(complete=True,
        video_language=args.world_video_language,noise_seed=args.policy_seed,
        intervention='Video text only; action text and all other production inputs retained.'),indent=2)+'\n')


if __name__=='__main__':main()
