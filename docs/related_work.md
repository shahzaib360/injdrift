# Related work

The gap this project targets sits between two well-studied lines of work: training
models to resist prompt injection, and the fragility of safety training under
downstream fine-tuning. As far as I could find (searched November 2026), nobody has
measured what an *ordinary, benign* downstream fine-tune does to the injection
robustness of a hardened model.

## Injection-hardened models (what gets eroded)

| work | what it does | why it matters here |
| --- | --- | --- |
| **StruQ** — Chen, Piet, Sitawarin, Wagner. *StruQ: Defending Against Prompt Injection with Structured Queries.* USENIX Security 2025. [arXiv:2402.06363](https://arxiv.org/abs/2402.06363) | Reserved delimiters + SFT on injected samples answered with the real task. | `injdrift harden --method struq` reproduces the recipe. |
| **SecAlign** — Chen, Zharmagambetov, Mahloujifar, Chaudhuri, Wagner, Guo. *SecAlign: Defending Against Prompt Injection with Preference Optimization.* ACM CCS 2025. [arXiv:2410.05451](https://arxiv.org/abs/2410.05451) | DPO on (injected prompt, secure answer, insecure answer); reports ~0% ASR for optimization-free attacks. | The hardening this project tries to erode; `injdrift harden --method secalign` is a small-scale replication. |
| **Meta SecAlign** — Chen, Zharmagambetov, Wagner, Guo. *Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks.* [arXiv:2507.02735](https://arxiv.org/abs/2507.02735) (2025; later versions retitled *Meta-SecAlign: Training LLMs against Prompt Injection for Robust Agents*) | Open-weight 8B/70B models with a dedicated `input` role for untrusted data; randomized injection positions; self-generated targets. | The "secure foundation model" whose guarantees a customer fine-tune would inherit — or not. `configs/grid/full.yaml` targets Meta-SecAlign-8B. |
| **Instruction Hierarchy** — Wallace, Xiao, Leike, Weng, Heidecke, Beutel. *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.* [arXiv:2404.13208](https://arxiv.org/abs/2404.13208) (2024) | Train models to rank system > user > tool instructions. | Same robustness-through-training premise, closed weights. |

## Attacks on those defenses (they are already brittle)

| work | finding |
| --- | --- |
| **ASTRA** — Pandya, Labunets, Gao, Fernandes. *May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks.* [arXiv:2507.07417](https://arxiv.org/abs/2507.07417) (2025) | An attention-redirecting optimization attack reaches high success against SecAlign and StruQ. This motivates the attention-share probe here. |
| **Surface heuristics** — Li, Yu, Ni, Li, Peris, Xiao, Zhao. *Defenses Against Prompt Attacks Learn Surface Heuristics.* [arXiv:2601.07185](https://arxiv.org/abs/2601.07185) (2026) | SFT-based defenses (incl. StruQ, SecAlign) learn position, token-trigger and topic shortcuts. If defenses are shallow, they should also be easy to wash out. This motivates the disjoint train/eval delimiter pools and the varied injection positions here. |
| **The Attacker Moves Second** — Nasr, Carlini, Sitawarin, Schulhoff, et al. [arXiv:2510.09023](https://arxiv.org/abs/2510.09023) (2025) | Adaptive attacks (gradient, RL, search, humans) bypass 12 recent jailbreak/injection defenses that reported near-zero ASR. This motivates reporting best-of-K and GCG next to static ASR. |
| **GCG** — Zou, Wang, Carlini, Nasr, Kolter, Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) (2023) | The optimization attack implemented in `injdrift.attacks.gcg`. |
| **Formalizing prompt injection** — Liu, Jia, Geng, Jia, Gong. *Formalizing and Benchmarking Prompt Injection Attacks and Defenses.* USENIX Security 2024. [arXiv:2310.12815](https://arxiv.org/abs/2310.12815) | Source of the "combined" attack family. |
| **Indirect prompt injection** — Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz. *Not what you've signed up for.* AISec 2023. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173) | The threat model: instructions arriving through data. |

## Fine-tuning erodes safety (the analogy)

| work | finding |
| --- | --- |
| **Qi et al.** — Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024. [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) | Benign fine-tuning (Alpaca, Dolly) measurably erodes *refusal* safety. This project asks the same question for *injection* robustness. |
| **Safety-Tuned LLaMAs** — Bianchi et al. ICLR 2024. [arXiv:2309.07875](https://arxiv.org/abs/2309.07875) | Mixing a small amount of safety data into fine-tuning preserves refusals → the `replay` baseline. |
| **Safe LoRA** — Hsu, Tsai, Lin, Chen, Yu, Huang. NeurIPS 2024. [arXiv:2405.16833](https://arxiv.org/abs/2405.16833) | Project LoRA updates using an alignment direction; a weight-space alternative to SecAnchor's loss-space anchoring (roadmap). |

## Instruction/data separation inside the model (the probes)

| work | relevance |
| --- | --- |
| **SEP** — Zverev, Abdelnabi, Tabesh, Fritz, Lampert. *Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?* ICLR 2025. [arXiv:2403.06833](https://arxiv.org/abs/2403.06833) | Defines instruction/data separation behaviourally; the role probe here is a representational counterpart with the same "same text, different role" control. |
| **ASIDE** — Zverev et al. *ASIDE: Architectural Separation of Instructions and Data in Language Models.* [arXiv:2503.10566](https://arxiv.org/abs/2503.10566) (2025) | Separates roles at the embedding level; suggests separation is a linearly readable property worth tracking. |
| **TaskTracker** — Abdelnabi, Fay, Cherubin, Salem, Fritz, Paverd. *Get my drift? Catching LLM Task Drift with Activation Deltas.* SaTML 2025. [arXiv:2406.00799](https://arxiv.org/abs/2406.00799) | Activation deltas at the last token detect injected tasks; the drift-ratio probe here adapts the idea to *track* a model across checkpoints. |

## Methods used

- **DPO** — Rafailov et al. 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
- **LoRA** — Hu et al. 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
