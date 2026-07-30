# NMI storyline & argument (locked 2026-07-02)

Positioning + contribution + rebuttals for the paper. Evidence inventory: `EXPERIMENTS.md`.
Data: `DATA_STATUS.md`. (This is a planning doc — prose is written locally by the author.)

---

## 0. One-line thesis
> Chemical reasoning models' chain-of-thought is **causally decoupled** from their answers — so it must
> **not** be trusted as reasoning to judge whether an answer is right — but it is the human's only
> **verifiable audit surface**; we prove the decoupling and give a verification-grounded framework that
> makes the audit report's **structural claims faithful** (fabrication ↓73–95%, no accuracy cost),
> turning CoT from a misleading pseudo-explanation into a trustworthy audit artifact.

**Title candidate:** *"Don't trust the reasoning, audit it: chemical reasoning models' chain-of-thought is
decoupled from their answers, and we make it a faithful audit report."*

---

## 1. The core logical chain (the spine)
1. **Decoupled** — corrupting the CoT's *content* flips only **1–11%** of correct answers; corrupting the
   *input* flips **40–79%**. The answer is input-dominant; the CoT content is not its cause.
2. **⇒ Dangerous to trust as reasoning** — you cannot use the CoT to judge whether the answer is correct;
   the "because X therefore Y" narrative is post-hoc and may be fabricated.
3. **But the CoT is still needed** — for two independent reasons, so removing it is not the fix:
   - **Model side (compute scaffold):** removing the CoT drops performance **20–53%** — the thinking-space
     provides serial computation. Its *content* is fungible (corrupting it barely hurts), its *presence*
     helps.
   - **Human side (audit surface):** the CoT is the *only* interface of checkable claims a chemist can
     inspect; with no CoT the model emits a molecule with zero auditable statements.
4. **⇒ Content is decorative to the ANSWER but load-bearing to the HUMAN.** A fabricated claim doesn't
   change the answer but *actively misleads the chemist who reads it*. Faithfulness is therefore an
   **oversight/trust axis, orthogonal to accuracy** (we hold accuracy, we don't improve it).
5. **⇒ Make the audit report faithful.** A decoupled-but-faithful report is a valuable oversight tool; a
   decoupled-and-fabricated one is worse than none. Our framework does this for the verifiable subset.

**The decoupling MOTIVATES the audit-report use (not undermines it):** *because* the CoT is not the
computation, it must be treated as a report to verify rather than reasoning to trust — and a report is only
useful if it is true.

---

## 2. Two-layer view of the CoT (scope of "faithful")
| Layer | Content | Machine-verifiable? | Our stance |
|---|---|---|---|
| **Structural-fact layer** | "has a hydroxyl / benzene ring / N atoms; valid molecule; target lacks group G" (the 2×2 ER/IO/EO) | ✅ RDKit, per claim | **We make it faithful** (ER ↓73–95%) |
| **Causal-narrative layer** | "adding a hydroxyl *therefore* raises solubility *so* choose X" | ❌ not checkable | decorative + non-causal → **should not be trusted as reasoning** |

We certify (and improve) the **verifiable structural claims** — the concrete, high-stakes subset a chemist
most directly relies on and is most misled by when fabricated. We do **not** claim to make the whole CoT
faithful. Honest scope, stated as a limitation.

---

## 3. Contributions
1. **Conceptual reframe (headline):** recast the CoT of scientific reasoning models from *explanation/reasoning*
   to *post-hoc audit report*; establish (with verifiable evidence) that it is not causal to the answer and
   must not be used to judge answer correctness, but can be made a trustworthy audit artifact. Novel: splits
   "faithfulness" into *causal reasoning* (unattainable, we prove) vs *verifiable-claim faithfulness*
   (attainable, useful).
2. **Verifiable measurement + evidence:** the only domain where reasoning claims are machine-checkable at
   scale (RDKit). Show (a) hallucination is pervasive & accuracy-invisible (decoupling); (b) causal analysis
   (perturbation Δlogp + drift conditional flip-to-wrong + entropy) proves CoT content is not the cause.
   Novel vs Turpin/Lanham (judge/behavior-based) and Sprague (black-box performance): ours is
   verifiable-per-claim + causal + in a high-stakes scientific domain.
3. **Method (deliverable):** turn the verifiable diagnoser into an online process reward → fabrication
   ↓73–95% at no accuracy cost, more verified claims (grounded term); coupled variant + anti-reward-hacking
   guardrail. Honest boundary: improves *report faithfulness* (oversight value), not *causal reliance* —
   and separating these is itself the insight.
4. **Origin / why standard training doesn't fix it:** fabrication is **not** installed by any single stage —
   the base model already fabricates most (per-claim **28.9%**, perf 0.3% = confabulation-under-ignorance);
   standard training leaves ~**19%** (even the released Chem-R); only explicit verification-grounded
   supervision reaches 2–3%. So training for accuracy alone cannot fix faithfulness (they are decoupled).

---

## 4. Evidence (real numbers, as of 2026-07-02)
- **Decoupling (drift, conditional flip-to-wrong among originally-correct):** corrupt CoT content
  (all_wrong) **1–11%**; drop CoT **20–53%** (OOD-confounded); corrupt input **40–79%**. Cross-model
  (Chem-R, ChemDFM, Faithful, coupled). **Retro holds** (content-corrupt ~7%, hardest task).
- **Entropy:** info_gain (H_noCoT − H_free) ≈ 0 → CoT gives ~no answer-uncertainty reduction. 3-condition
  cot_condsent (empty/real/corrupted) running for the metric-free confirmation (ig_presence vs ig_content).
- **No re-coupling:** training makes models *more* input-grounded, not less. (drop the "re-couple" claim.)
- **Mitigation:** ER ↓73–95% per family, perf held/↑, %ER=0 ↑, claim precision & verified-FG count ↑.
- **Origin ladder:** base-a 28.9% per-claim fab (perf 0.3%) → off-the-shelf Chem-R 19.4% → process 3.4% →
  coupled 2.1%. (SFT row pending.)
- **ether-0 exception:** relies on CoT presence (drop flips 44–53%), ignores input — differently trained,
  not on these tasks ⇒ no internalized shortcut ⇒ must reason. So decoupling is a property of task-RL'd
  models, not a universal law (state as scope, not overclaim).

---

## 5. Rebuttals to expected reviewer objections
- **"If the CoT is decoupled/useless, why bother making the report faithful — isn't it redundant?"**
  Content is decorative *to the answer* but load-bearing *to the human who reads it*. A fabricated claim
  misleads the chemist even though it doesn't change the answer. Faithfulness protects the human (oversight),
  not the metric. And the CoT can't be dropped (compute scaffold: −20–53%; and it's the only audit surface).
- **"CoT unfaithfulness is known (Turpin, Lanham)."** They observe it via judges/behavior; we *verify it
  per claim against ground truth*, *locate its origin*, and *reduce it* — in a domain with real stakes.
- **"Then just skip the CoT and output the answer."** Performance drops 20–53% (compute), and you lose the
  audit surface entirely. Keep the CoT; make it faithful; teach correct use (verify, don't trust).
- **"You only check functional groups, not real reasoning."** Correct and stated: we scope to
  verifiable structural claims — the concrete, high-risk subset — and explicitly do not certify the causal
  narrative (which we separately show is decoupled and shouldn't be trusted anyway).
- **"Isn't this just S2 metric gaming / does faithfulness cost accuracy?"** Accuracy held/improved; reward
  guarded against hacking (official success, FG de-dup by category); coupled variant.

---

## 6. Danger / impact framing (for intro & discussion)
In molecular design / retrosynthesis / drug discovery, chemists read the model's step-by-step chemical
reasoning as justification for its outputs. If they judge an answer by that reasoning, they trust a story
that (a) may be fabricated and (b) is not what produced the answer — so wrong molecules get adopted for
wrong reasons, and the reasoning "explains away" errors that go uncaught. Safe practice we establish: treat
CoT as an audit report to be **verified** (not as justification), pair it with a verification layer (our
diagnoser), and train models to make the report faithful.

---

## 7. Detector validity — WHY no human study is needed
The ER verdict is **exact RDKit SMARTS substructure matching**, not an LLM-judge or heuristic: "fabricated"
= the claimed functional group is *provably absent* by exact chemistry. It is correct **by construction**, so
per-claim human annotation would only re-confirm SMARTS — there is no judgment to validate. (This is a
strength: unlike general-LLM hallucination work that must rely on judges/annotators, our ground truth is
deterministic.) The one non-exact step is **claim extraction** (parsing which FGs the CoT asserts, from text);
if challenged, cover it with a lightweight extractor-precision spot-check, not a chemist study. → per-claim
human validation experiment is DROPPED. (The blind arena, a separate trace-level human-preference result, stays.)

## 8. Still needed (priority order)
1. condsent 3-condition entropy (running) — metric-free confirmation of content-decorative.
2. SFT ladder row (running) — complete the origin curve.
3. Optional: filler-token condition (COCONUT/latent-compute framing: content vs compute-space); s2
   similarity-weighted success (edit/opt scaffold) as a success-metric fallback.
