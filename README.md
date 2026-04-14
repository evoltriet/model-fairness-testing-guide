# Model Fairness Testing Guide

Notebook-driven guide for auditing model behavior across sensitive groups and comparing fairness mitigation strategies in practice.

## Problem
The repo explores how to test a model for bias, explain the observed tradeoffs, and decide whether mitigation is actually appropriate for the use case. The example domain is dental claims review, where the notebook frames the model as an x-ray-based approval system that can still be routed to a human reviewer when the model is uncertain.

This is deliberately closer to an AI review-board / responsible-AI proof of work than a polished library or production service.

## Approach
The repository contains two complementary notebooks:

- [`example_fairness_testing.ipynb`](example_fairness_testing.ipynb) walks through the dental use case end to end, including group analysis by gender, race, age, geography, and other features.
- [`fairness_toolings_experiment.ipynb`](fairness_toolings_experiment.ipynb) compares fairness tooling on the Adult income dataset and demonstrates intervention techniques such as `ExponentiatedGradient` with a demographic-parity objective.

The repo also includes helper code in [`utils/fairness_intervention.py`](utils/fairness_intervention.py), which sweeps thresholds and evaluates profit-versus-fairness tradeoffs across groups.

## Outcome
The notebook outputs make a few points clear:

- The dental use case emphasizes minimizing false positives because failed approvals are sent to a human reviewer.
- For gender analysis, the notebook uses imputed gender as an example when self-reported data is unavailable, while noting that self-reported data is preferred.
- For age analysis, the under-18 group shows a higher false positive rate, roughly `1.25%` in the notebook output.
- The notebook explicitly notes that demographic parity mitigation can reduce selection-rate gaps, but equalized odds may be too costly for this use case because the underlying false positive rate is already low.
- The tooling notebook surfaces feature dependence signals, with examples such as `education-num`, `fnlwgt`, `race_White`, and `sex_Male` showing up in the FairML-style ranking.

The main outcome is not "perfect fairness," but a structured process for identifying tradeoffs and explaining them in plain language.

## How To Explore
- Start with [`example_fairness_testing.ipynb`](example_fairness_testing.ipynb) for the dental model story.
- Then open [`fairness_toolings_experiment.ipynb`](fairness_toolings_experiment.ipynb) to compare fairness libraries and mitigation strategies.
- Check [`utils/fairness_intervention.py`](utils/fairness_intervention.py) for the threshold-sweep logic used in the intervention comparison.
- The notebook references an embedded chart in [`images/fairness-tradeoffs.png`](images/fairness-tradeoffs.png) that helps explain the profit/fairness tradeoff visually.

## Limitations
- The repo is notebook-first and depends on the data shape and environment used during experimentation.
- Some examples rely on imputed demographic data, which the notebook itself flags as a fallback, not the preferred source.
- The results are use-case specific and should be read as exploratory fairness analysis, not as a universal fairness benchmark.
- The mitigation notebook uses profit functions and threshold sweeps, so the "best" intervention depends on the business objective and the acceptable risk threshold.
