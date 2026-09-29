"""
scripts/finetune_lora.py

LoRA fine-tuning for PII memorization (Day 5, Framing B).

목적
----
Pre-trained Gemma-2-9B / Llama3-Med42-8B는 우리 600 합성 환자 PII를 본 적 없음.
Day 4 sanity check: pretrained 모델이 이 PII를 외운 적이 없어 intervention의
"PII 차단 효과"를 측정할 baseline이 약했음.

이 스크립트는 train 600 P_L에 대해 LoRA fine-tune을 돌려 모델이 PII를
"외우게" 만든 다음, intervention 효과를 측정할 baseline을 만듦.

검증 기준 (이 스크립트 단독으론 측정 안 함, evaluate_extraction.py에서 함)
  - train 600 extraction rate ≥ 90%   → memorization 성공
  - test  60 extraction rate < 20%    → 일반화 아닌 암기
이 두 조건이 갈라지면 "외웠고 일반화 아님" → Day 6+ intervention의 baseline.

사용
----
  python scripts/finetune_lora.py --model gemma_2_9b
  python scripts/finetune_lora.py --model llama3_med_8b --batch 8

GPU
---
RTX PRO 6000 Blackwell 97GB × 2장. device_map="auto"로 모델을 GPU에 분산.
LoRA는 학습 파라미터가 적어 (~0.5%) batch 4~8까지 시도 가능.
"""

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

# config.MODEL_CONFIGS / DATA_DIR 등을 가져오기 위한 path 등록
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config


# ── 1. CLI 인수 ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(config.MODEL_CONFIGS), required=True,
                   help="config.MODEL_CONFIGS의 키 (gemma_2_9b / llama3_med_8b)")

    # Optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=4,
                   help="per-device batch size (effective = batch * grad_accum)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=50)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank — 낮을수록 가볍고 학습량 적음")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA scaling — 관례적으로 r * 2")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--include-mlp", action="store_true",
                   help="Add MLP layers (gate_proj/up_proj/down_proj) to LoRA targets — needed when attention-only LoRA can't overcome RLHF prior.")

    # Data / length
    p.add_argument("--max-length", type=int, default=512,
                   help="P_L 최대 327 token이라 512면 안전")
    p.add_argument("--pairs-train", default=os.path.join(config.DATA_DIR, "pairs.jsonl"))
    p.add_argument("--registry-train",
                   default=os.path.join(config.DATA_DIR, "profile_registry.jsonl"))

    # Early-stop
    p.add_argument("--early-stop-extraction", type=float, default=0.90,
                   help="train extraction rate가 이 값에 도달하면 학습 종료")
    p.add_argument("--eval-samples", type=int, default=30,
                   help="매 epoch 끝에서 extraction 측정에 쓸 sample 수")

    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-base", default=os.environ.get("CHECKPOINT_DIR", "/path/to/checkpoints"),
                   help="checkpoint 저장 root")

    return p.parse_args()


# ── 2. Dataset ────────────────────────────────────────────────────────────
# 600 P_L을 next-token prediction format으로 들고 있는 Dataset.
#
# Causal LM의 표준 fine-tune:
#   - input  = full P_L 토큰
#   - target = 같은 토큰 (모델 내부에서 1-token shift돼서 cross-entropy 계산)
#   - loss는 모든 non-padding 토큰에 적용 (PII만 mask하지 않음 — 표준 SFT)
#   - padding 위치는 label=-100으로 두면 cross-entropy에서 자동 무시됨

class PIIDataset(Dataset):
    """pairs.jsonl을 읽어 P_L만 추출, max_length로 padding/truncate."""

    def __init__(self, pairs_path: str, tokenizer, max_length: int):
        self.examples: list[str] = []
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                self.examples.append(pair["p_l"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # squeeze(0): tokenizer는 [1, L] 반환 → [L]로
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)

        # labels: input_ids 그대로 (HF가 내부에서 1-token shift해서 loss 계산).
        # padding 위치는 -100으로 두면 loss 계산에서 자동 무시.
        labels = ids.clone()
        labels[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ── 3. 모델 + LoRA 부착 ────────────────────────────────────────────────────
def build_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float,
                 include_mlp: bool = False):
    """Pre-trained 모델 + tokenizer 로드, LoRA adapter 부착해서 반환.

    LoRA 메커니즘
    ------------
    원본 weight W (예: q_proj.weight, [d, d])는 동결 (학습 X).
    옆에 작은 low-rank 행렬 A ([d, r]), B ([r, d])만 학습 가능하게 추가.
    Forward 시 효과적인 weight: W + (alpha/r) * A @ B.
    학습 파라미터 수가 ~0.5%로 줄어 메모리/속도 절약.

    target_modules = q_proj, k_proj, v_proj, o_proj
      attention block의 4개 projection. Gemma-2와 Llama-3 모두 모듈 이름 동일.
      (model.named_modules()로 grep해서 확인 가능)
    """
    cfg = config.MODEL_CONFIGS[model_name]
    model_path = cfg["path"]

    print(f"[build_model] loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        # Llama 계열은 pad_token이 없으므로 eos를 재사용. Gemma는 이미 있어서 영향 X.
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[build_model] loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",   # 자동 GPU 분산 (RTX PRO 6000 × 2)
    )

    # gradient_checkpointing: forward에서 activation을 저장하지 않고 backward에서
    # 다시 forward를 돌려 activation 재계산 → 메모리 절약, 속도 약간 ↓.
    model.gradient_checkpointing_enable()
    # PEFT + gradient_checkpointing 같이 쓸 때는 input embedding이 gradient를
    # 받아야 backward 그래프가 LoRA 파라미터까지 연결됨. (frozen embedding이라
    # 기본은 grad 안 받음 → 이걸로 강제 설정)
    model.enable_input_require_grads()

    # use_cache: generation에서 KV 캐시. 학습 중에는 backward와 비호환이라 False.
    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    print(f"[build_model] LoRA target_modules = {target_modules}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 보통 0.3~0.5% trainable이라고 출력됨

    return model, tokenizer


# ── 4. extraction rate 측정 ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_extraction(
    model,
    tokenizer,
    pairs_path: str,
    registry_path: str,
    n_samples: int = 30,
    indices: list[int] | None = None,
    pii_field: str = "full_name",
    marker: str = "Patient Name: ",
    max_new_tokens: int = 20,
) -> float:
    """sample에서 prompt만 주고 generate, 정답 PII가 등장하는지 substring match.

    절차
    ----
    1. pairs.jsonl과 registry.jsonl을 entity_id로 join → ground-truth 가져옴.
    2. P_L에서 marker (예: "Patient Name: ") 위치까지가 prompt.
       그 뒤를 모델이 생성 → 외운 PII가 나오는지 확인.
    3. greedy decoding (do_sample=False) — 재현성 + memorization 측정에 적합.
    4. 정답 (예: "James Smith")이 생성 텍스트에 포함되면 hit.

    pii_field / marker를 바꾸면 다른 PII 종류 (id/phone/email)도 측정 가능.
    이 callback에서는 빠른 sanity check용으로 name만 측정.

    indices가 주어지면 그 인덱스 pair만 사용 (epoch 간 재현성 보장).
    None이면 random.sample(n_samples).
    """
    # entity_id → registry record 매핑
    registry: dict[str, dict] = {}
    with open(registry_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            registry[r["entity_id"]] = r

    pairs: list[dict] = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    if indices is not None:
        sample = [pairs[i] for i in indices if 0 <= i < len(pairs)]
    else:
        sample = random.sample(pairs, min(n_samples, len(pairs)))

    was_training = model.training
    model.eval()

    correct = 0
    measured = 0
    for pair in sample:
        p_l = pair["p_l"]
        marker_idx = p_l.find(marker)
        if marker_idx < 0:
            continue  # marker 없으면 skip (정상 케이스에선 발생 X)

        # marker까지 포함한 prefix가 prompt → 그 뒤를 모델이 생성
        prompt = p_l[: marker_idx + len(marker)]
        target = registry[pair["entity_id"]][pii_field]

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # PII는 prompt 직후 생성되므로 prompt 잘림은 측정에 영향 X
        ).to(model.device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                        # greedy
            pad_token_id=tokenizer.pad_token_id,
        )
        # generate 결과는 [prompt + generated]. prompt 길이만큼 자르고 디코딩.
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if target in gen_text:
            correct += 1
        measured += 1

    if was_training:
        model.train()

    return (correct / measured) if measured > 0 else 0.0


# ── 5. epoch 끝에서 extraction 측정하는 callback ─────────────────────────────
class ExtractionCallback(TrainerCallback):
    """매 epoch 끝에서 train extraction rate를 측정하고,
    early_stop_threshold 도달 시 control.should_training_stop=True를 set.

    HF Trainer는 매 epoch 끝에 on_epoch_end(args, state, control, **kwargs)를 호출.
    kwargs에는 model, tokenizer 등이 들어옴.
    control은 학습 흐름 제어 객체 (should_training_stop를 True로 set하면 종료).
    """

    def __init__(
        self,
        tokenizer,
        pairs_path: str,
        registry_path: str,
        n_samples: int,
        early_stop_threshold: float,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.pairs_path = pairs_path
        self.registry_path = registry_path
        self.n_samples = n_samples
        self.threshold = early_stop_threshold
        self.history: list[dict] = []  # {epoch, rate} 누적

        # epoch 간 비교 가능하도록 sample을 한 번만 결정.
        # __init__ 시점에 pairs.jsonl을 한 번 읽어 length 파악, seed 고정 RNG로
        # 인덱스 추출. global random.seed에 의존하지 않음.
        with open(pairs_path, encoding="utf-8") as f:
            n_pairs = sum(1 for line in f if line.strip())
        rng = random.Random(seed)
        self.fixed_indices = rng.sample(range(n_pairs), min(n_samples, n_pairs))
        print(f"[ExtractionCallback] fixed {len(self.fixed_indices)} eval indices "
              f"(seed={seed}, total {n_pairs} pairs)")

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        rate = evaluate_extraction(
            model,
            self.tokenizer,
            pairs_path=self.pairs_path,
            registry_path=self.registry_path,
            indices=self.fixed_indices,   # 매 epoch 동일 sample
        )
        epoch = int(state.epoch) if state.epoch is not None else -1
        self.history.append({"epoch": epoch, "name_extraction_rate": rate})
        print(f"[ExtractionCallback] epoch {epoch}: train name extraction = {rate:.2%}")

        if rate >= self.threshold:
            print(f"[ExtractionCallback] threshold {self.threshold:.0%} reached → "
                  f"setting should_training_stop")
            control.should_training_stop = True
        return control


# ── 6. main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 재현성: random/numpy/torch seed 모두 set
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = os.path.join(args.output_base, f"{args.model}_lora")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = build_model(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_mlp=args.include_mlp,
    )

    train_dataset = PIIDataset(args.pairs_train, tokenizer, args.max_length)
    print(f"[main] train dataset: {len(train_dataset)} examples (max_length={args.max_length})")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_strategy="epoch",        # 매 epoch 끝에서 LoRA adapter 저장
        save_total_limit=2,           # 디스크 절약 (마지막 2개만 유지)
        bf16=True,
        report_to="none",             # wandb / tensorboard 비활성
        seed=args.seed,
        remove_unused_columns=False,  # PIIDataset은 input_ids/mask/labels만 반환
        dataloader_num_workers=2,
    )

    extraction_cb = ExtractionCallback(
        tokenizer=tokenizer,
        pairs_path=args.pairs_train,
        registry_path=args.registry_train,
        n_samples=args.eval_samples,
        early_stop_threshold=args.early_stop_extraction,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[extraction_cb],
    )

    print(f"[main] starting training: {args.model}, "
          f"effective_batch={args.batch * args.grad_accum}, "
          f"epochs<={args.epochs}, lr={args.lr}")
    trainer.train()

    # 최종 LoRA adapter 저장 (epoch checkpoint와 별도로 final/에 깔끔히)
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[main] saved adapter to {final_dir}")

    # extraction history 저장 (학습 곡선 분석용)
    history_path = os.path.join(output_dir, "extraction_history.json")
    with open(history_path, "w") as f:
        json.dump(extraction_cb.history, f, indent=2)
    print(f"[main] saved extraction history to {history_path}")


if __name__ == "__main__":
    main()
