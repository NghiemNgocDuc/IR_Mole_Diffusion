import json
import numpy as np
import pytest
from test_compatibility import tiny_experiment, single_thread
from diffusion.generated_matcher_data import generated_plan
from diffusion.data import file_sha256
import train_generated_matcher as runner


def test_generated_negatives_remove_false_negatives_and_fill_distinct_fallbacks():
    # Duplicate identity at another bank index must also be excluded.
    plan, stats = generated_plan(np.array([[1,2,3],[0,2,3]]), [[4,5,5], []],
                                [0,1], ['C','N','O','F','C','CC'])
    assert plan.tolist() == [[5,1,2],[0,2,3]]
    assert stats['generated_slots'] == 1 and stats['fallback_slots'] == 5
    with pytest.raises(ValueError):
        generated_plan(np.array([[1,2,3]]), [[99]], [0], ['C','N','O','F'])


def test_real_controlled_pipeline(tiny_experiment, single_thread, monkeypatch):
    _, prior, checkpoint, report = tiny_experiment
    output = prior.parent/'generated_matcher'
    protected = {p:file_sha256(p) for p in (checkpoint, report)}
    score = runner.score_pools
    observed = []
    def check(*pos, **kw):
        label = kw.get('label',pos[-1])
        if label.startswith('assessment:'):
            selection = json.loads((output/'selection.json').read_text())
            assert set(selection['arms']) == {'original','generated'}
            assert not selection['assessment_used_for_selection']
            observed.append(label)
        return score(*pos,**kw)
    monkeypatch.setattr(runner,'score_pools',check)
    args = ['--checkpoint',str(checkpoint),'--baseline_report',str(report),'--output_dir',str(output),
        '--data',str(prior.parent/'data.npz'),'--splits',str(prior.parent/'splits.json'),
        '--device','cpu','--epochs','2','--warmup_epochs','1','--batch_size','4','--dev_size','1',
        '--training_limit','5','--matcher_layers','1','--progress_interval','999','--diagnostic']
    result = runner.main(args)
    assert observed and not result['weights_promoted']
    cfg=json.loads((output/'configuration.json').read_text())
    assert len(cfg['gradient_training_rows']) == 5
    assert set(cfg['gradient_training_rows']) <= set(range(6))
    arms=result['selection']['arms']
    assert arms['original']['initial_state_sha256'] == arms['generated']['initial_state_sha256']
    for arm in runner.ARMS:
        assert arms[arm]['final_encoder_sha256'] == cfg['encoder_start_sha256']
        history=json.loads((output/arm/'history.json').read_text())
        assert all(e['training_rows']==5 and not e['encoder_trainable'] for e in history)
        for mode in result['arms'][arm].values():
            assert mode['summary']['attempted_rows']==2
            assert mode['summary']['top5_identity_fraction']==result['baseline']['top5_identity_fraction']
    assert all(file_sha256(p)==v for p,v in protected.items())
    assert (output/'generated_matcher_results.zip').is_file()
    with pytest.raises(SystemExit):
        runner.main(args)
