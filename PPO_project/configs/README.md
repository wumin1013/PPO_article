# Config Layout

`configs/` has been simplified to three active profiles:

- `default.yaml`: current mainline method (learnable lookahead + cornerness-aware reward).
- `train_square_p0.yaml`: `P0` baseline.
- `p0_l2_gold.yaml`: `P0_gold` baseline.

Legacy/ablation/temporary configs are archived under:

- `configs/archive/legacy/`

## Quick Use

Mainline training:

```powershell
python main.py --mode train --config configs/default.yaml
```

P0 baseline training:

```powershell
python main.py --mode train --config configs/train_square_p0.yaml
```

P0_gold baseline training:

```powershell
python main.py --mode train --config configs/p0_l2_gold.yaml
```
