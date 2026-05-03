# Seeing Clearly: When Should an Emotion Classifier Speak?

*A technical blog post based on the Seeing Clearly repository, proposal, final reproducibility notebook, and final figures.*

## Abstract

Facial-expression recognition is often framed as a straightforward image-classification problem, but that framing breaks down quickly when the output is meant to support real social interaction. *Seeing Clearly* explores this tension through an assistive webcam prototype that predicts one of seven FER-2013 expression labels and surfaces those predictions as soft cues rather than hard judgments. The project combines transfer learning with a ResNet-18 backbone, targeted analysis of class imbalance, and uncertainty-aware interface design. The final system does not fully meet its original benchmark hypothesis of 70% test accuracy, but it comes close: the selected clean transfer model reaches 68.56% accuracy on the held-out FER-2013 test set and 69.55% validation accuracy during model selection. More importantly, the project shows that aggregate accuracy alone is the wrong success criterion for an assistive setting. The most useful result is not simply that the classifier works moderately well, but that reliability varies sharply by class and that confidence thresholds can trade coverage for trustworthiness. This makes the project a useful case study in the difference between a benchmark model and a socially safer interface.

## 1. Introduction

Facial expressions carry information that many people use automatically in conversation. For some users, however, especially people who experience difficulty interpreting facial cues, those signals can be noisy, stressful, or easy to misread. The original proposal for *Seeing Clearly* framed the project as an assistive facial-expression recognition system: a tool that could act less like a mind reader and more like a second pair of eyes. That framing matters. In an accessibility context, the design goal is not to produce dramatic claims about hidden internal states. It is to provide cautious, visibly grounded cues that may help a user notice patterns they might otherwise miss.

That distinction shapes the machine learning problem. A conventional classifier must always output a class, but an assistive system does not have to present every prediction with equal force. If the model is uncertain, poorly calibrated, or looking at an ambiguous expression, the interface can hedge, delay, or stay quiet. This makes the central question of the project more subtle than “How accurate is the model?” The better question is: **When is a transfer-learned facial-expression classifier reliable enough to surface a cue, and when should the interface withhold or soften its output?**

This blog post analyzes the final artifacts in the *Seeing Clearly* repository through that lens. It uses the final reproducibility notebook, saved experiment metadata, and generated figures to tell a clearer technical story than a short proposal or demo alone can provide.

## 2. Hypothesis

The proposal stated a concrete initial hypothesis: a fine-tuned ResNet-18 using transfer learning would achieve at least 70% accuracy on FER-2013 while remaining lightweight enough for real-time webcam assistance. That hypothesis was useful because it imposed a measurable target and encouraged a model class that could plausibly run in an interactive application.

For the final project, the hypothesis can be sharpened into two parts:

1. **Performance hypothesis:** A transfer-learned ResNet-18 trained on FER-2013 can approach 70% held-out accuracy while outperforming random chance by a large margin.
2. **Interface hypothesis:** In an assistive setting, uncertainty-aware presentation strategies such as confidence thresholding and temporal smoothing make the system more useful than always displaying the top predicted class.

The final results partially support both claims. The model comes close to the accuracy target but does not clearly exceed it. At the same time, the uncertainty analysis strongly supports the second claim: confidence carries useful information, and forcing the system to “speak” on every frame would be less responsible than letting it hedge.

## 3. Background and Related Work

The project sits at the intersection of three strands of prior work: social-cognitive motivation, benchmark facial-expression recognition, and confidence calibration.

The social motivation comes from work such as Baron-Cohen et al.’s revised “Reading the Mind in the Eyes” test, which examines how adults infer mental states from facial cues and shows that this kind of social sensitivity varies meaningfully across people and groups. That work does **not** imply that emotions can be read directly from faces by a model. What it does justify is the broader idea that facial cues matter in social interpretation and that difficulty reading them can be consequential in everyday interaction.

The computer-vision side of the project relies on FER-2013, introduced through the ICML 2013 representation learning challenges reported by Goodfellow et al. FER-2013 remains popular because it offers a manageable benchmark with over 35,000 grayscale facial images labeled into seven expression classes. It is also a difficult dataset for exactly the reasons that make it interesting: the images are only 48x48 pixels, labels are coarse, some expressions are visually ambiguous, and the class distribution is imbalanced. A model that performs “well” on FER-2013 is still operating under strong representational constraints.

Architecturally, the project uses ResNet-18, drawing on the residual-learning framework introduced by He et al. Residual connections made it practical to train substantially deeper networks by reformulating layers around residual functions rather than unconstrained mappings. For *Seeing Clearly*, ResNet-18 is a sensible compromise. It is expressive enough to transfer useful visual features from ImageNet pretraining, but still lightweight enough for a webcam prototype on consumer hardware.

Finally, the project’s uncertainty framing is motivated by Guo et al.’s work on calibration in modern neural networks. Their paper shows that strong classifiers can still be poorly calibrated: confidence scores may not correspond well to empirical correctness. In an assistive interface, that matters a great deal. A mildly wrong model that speaks cautiously may be safer than a more accurate model that is confidently wrong.

Taken together, the literature suggests a productive but constrained design space: use a strong transferable visual backbone, train on a standard benchmark, but evaluate the resulting system in terms of not just accuracy, but class-level behavior and confidence reliability.

## 4. Repository and Project Scope

One of the more interesting aspects of the repository is that it captures the project’s evolution. The earlier branch preserves a fuller web-app and training-script structure, while the final GitHub submission narrows the project to a reproducibility notebook, saved model metadata, generated figures, and a technical write-up. That final organization is a strength: it makes the core claims easier to audit.

The two most important final artifacts are:

- `notebooks/Seeing_Clearly_Final_Reproducibility.ipynb`
- `index.html`

The notebook handles dataset loading, experiment execution, checkpoint selection, evaluation, and figure generation. The static blog post translates those results into a narrative about reliability and assistive use. A separate `app.py` implements a lightweight Streamlit webcam prototype that performs face detection, applies the classifier, smooths probabilities across frames, and turns predictions into gentle conversational prompts.

## 5. Methods

### Dataset and preprocessing

The project uses FER-2013, which contains grayscale face crops labeled as `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, or `surprise`. Because ResNet-18 expects three-channel images at higher spatial resolution than 48x48, the preprocessing pipeline converts grayscale inputs into three channels and resizes them to 224x224. During training, the notebook applies augmentation including random crops, flips, rotations, affine transforms, contrast adjustments, and random erasing. These choices are reasonable for a low-resolution dataset where overfitting is a serious risk.

![Figure 1. One FER-2013 example from each class.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/fer2013_examples.png)

**Figure 1.** One FER-2013 training example from each class. The low resolution and class ambiguity are part of the learning problem, not just a preprocessing nuisance.

### Model architecture

The classifier uses ResNet-18 with a modified head. The final fully connected layer is replaced by a dropout layer followed by a seven-class linear classifier. This is a standard but effective transfer-learning pattern: keep a strong pretrained feature extractor, then adapt the final decision layers to the target task. In the live app, the model is paired with OpenCV Haar-cascade face detection and a preprocessing step that applies CLAHE to normalize contrast on detected face regions.

### Experimental design

The notebook distinguishes between exploratory experiments and a cleaner final protocol. This is an important methodological improvement over a simple “train a few models and keep the best-looking one” workflow. Some earlier continuation experiments were informative but not ideal as final evidence because they involved checkpoints with more complicated histories. The final comparison instead fixes a single stratified train/validation split and evaluates all candidate models under that shared setup.

The main candidates were:

| Candidate | Best epoch | Validation accuracy | Macro accuracy | Weak-class accuracy | Selection score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean transfer baseline | 11 | 69.55% | 66.41% | 57.05% | 67.98% |
| Balanced, weak boost 1.2 | 14 | 68.69% | 67.61% | 57.94% | 66.11% |
| Balanced, weak boost 1.4 | 14 | 67.72% | 66.78% | 58.17% | 65.44% |

**Table 1.** Final clean-model comparison from the repository’s selected experiment metadata. The balanced variants helped weaker classes slightly, but the plain transfer baseline won the final selection criterion.

This table captures one of the project’s most useful findings. Class-balanced strategies did improve some weak classes, especially those that are harder and more socially consequential, but they did not produce the best overall clean selection score. The final model is therefore not the most complicated one. It is the simplest candidate that held up best under the chosen protocol.

### Prototype interface

The live prototype adds one more layer of applied ML design. Predictions are smoothed across recent frames using a short temporal average, which helps reduce label flicker. The interface then turns predicted expressions into prompts such as “Ask if everything is okay” rather than displaying emotionally loaded assertions. That distinction is subtle but important. The UI design encodes uncertainty and social caution into the product itself.

## 6. Results

The selected final model reaches **68.56% accuracy on the FER-2013 held-out test set**. That result falls just short of the original 70% benchmark hypothesis, but it still represents a strong improvement over early project runs and a substantial gain over random chance. It supports the claim that transfer learning with ResNet-18 is a viable baseline for this task.

![Figure 2. Training history for the selected model.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/training_history.png)

**Figure 2.** Training history for the selected clean model. Validation performance approaches the original target without clearly surpassing it.

The training curves suggest a reasonably healthy fit rather than catastrophic overfitting. Accuracy improves steadily, and the selected epoch is chosen on validation behavior rather than simply the last epoch. This is reassuring, but the more interesting story appears when the results are broken down by class.

![Figure 3. Row-normalized confusion matrix.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/confusion_matrix_normalized.png)

**Figure 3.** Row-normalized confusion matrix on the held-out FER-2013 test set.

The confusion matrix shows that performance is uneven. `happy` and `surprise` are easier for the model, while classes such as `fear` and `sad` remain harder. That matters more than it would in a generic benchmark setting because negative-expression confusions are exactly the kind of errors that can make an assistive system misleading. If a model frequently confuses fear, sadness, and anger, the consequences are qualitatively different from confusion between happier expressions.

![Figure 4. Per-class accuracy.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/per_class_accuracy.png)

**Figure 4.** Per-class accuracy emphasizes that the model is not uniformly reliable across expression categories.

The per-class breakdown makes the same point more directly: this is not a universal emotion reader. It is a model with strong and weak zones. That makes overall accuracy a necessary but insufficient metric for deployment-minded evaluation.

The uncertainty analysis is where the project becomes most compelling. Instead of assuming that every argmax prediction should be shown, the notebook measures how accuracy changes as the confidence threshold rises. The result is an interpretable coverage-versus-reliability tradeoff.

![Figure 5. Confidence threshold tradeoff.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/confidence_threshold_tradeoff.png)

**Figure 5.** Raising the confidence threshold reduces coverage but improves the reliability of the predictions that remain.

This figure strongly supports the second hypothesis. An assistive interface does not need to label every frame. If it only surfaces predictions above a confidence threshold, it can become more trustworthy at the cost of covering fewer situations. In a classroom leaderboard context that may seem like a compromise. In a human-facing assistive context, it is arguably the correct design choice.

The repository also includes a reliability diagram and reports an expected calibration error of about **0.067** for the selected model. That number is not perfect, but it suggests that confidence scores are informative enough to use as part of the interface logic. In other words, confidence is not just decorative. It contains meaningful signal about when the model should hedge.

Overall, the results support a nuanced conclusion:

- The project nearly hits its performance target.
- The baseline transfer model is stronger than more complex balanced variants under the final protocol.
- Aggregate accuracy overstates how reliable the system is in practice.
- Confidence-aware filtering and interface caution are not optional extras; they are central to the system’s usefulness.

## 7. Limitations

The project has several limitations, and the repository is strongest when it acknowledges them directly.

The first limitation is conceptual. FER-2013 labels are not emotions in the deep psychological sense. They are dataset categories attached to static images. A classifier trained on those labels is learning to map face crops to benchmark classes, not to infer a person’s internal emotional reality. The blog post in the repository is careful about this distinction, and that caution is justified.

The second limitation is dataset quality. FER-2013 is low-resolution, visually noisy, and imbalanced. While those properties make it a legitimate challenge dataset, they also limit what can be concluded from a model that performs well on it. Strong results on FER-2013 do not guarantee robustness across lighting conditions, ages, cultures, camera qualities, or expression styles.

The third limitation is evaluation scope. The final analysis is thorough within the benchmark setting, but it is still benchmark-centric. The webcam prototype uses temporal smoothing and prompt wording to improve usability, yet there is no human-subject evaluation showing whether the interface actually helps users, distracts them, or changes behavior in beneficial ways.

Finally, there is branch-level reproducibility nuance. The repository’s final analysis artifacts are clear, but some model weights are stored through Git LFS and some earlier experiments remain exploratory rather than fully standardized. That does not invalidate the project, but it does mean the clean final protocol should be treated as the authoritative evidence rather than every historical experiment in the repo.

## 8. Ethics and Responsible Use

The ethical issues here are not secondary; they are part of the technical problem definition.

First, facial-expression classification risks overstating what is observable from a face. Expressions are shaped by culture, personality, masking, context, disability, fatigue, and social norms. An assistive tool that presents outputs as facts could easily encourage overconfidence in weak inferences. The project makes the right move by describing predictions as “soft cues” and pairing them with prompts rather than blunt declarations.

Second, fairness remains unresolved. The repository does not contain the demographic annotations needed for a meaningful subgroup analysis. That means the model could perform unevenly across race, age, gender presentation, disability, or other axes without the current evaluation detecting it. In a real deployment, that would be a major concern.

Third, privacy and consent matter. A local classroom prototype that runs on-device is ethically very different from a system embedded in broader surveillance or workplace monitoring. Tools for facial analysis are especially vulnerable to misuse once they leave their original intended context.

For those reasons, the ethically strongest version of *Seeing Clearly* is not a high-confidence emotional truth machine. It is a local, transparent, consent-based prototype that explicitly communicates uncertainty, avoids making mental-state claims, and treats the human user’s own judgment as primary.

## 9. Conclusion

*Seeing Clearly* is most successful when read not as a claim that webcams can read emotions, but as a careful study of what happens when a facial-expression classifier is pulled toward assistive use. The final ResNet-18 model nearly reaches the original performance target, and the project demonstrates that transfer learning can produce a usable FER-2013 classifier for real-time prototyping. But the deeper result is methodological: for human-facing ML systems, benchmark accuracy is only the beginning of the story.

The repository’s best contribution is its shift from “What is the top-1 label?” to “When should the system speak?” Once that question is asked, per-class errors, calibration, thresholding, and interface language become central design decisions rather than afterthoughts. That makes *Seeing Clearly* a thoughtful applied ML project: technically competent, modest in its claims, and strongest where it admits the limits of what the model can know.

If this work were extended, the next most valuable steps would be temperature scaling, quantitative evaluation of temporal smoothing, a more diverse dataset, and real user studies focused on whether the prompts are genuinely supportive. Those directions would move the project from a promising class project toward a more credible assistive research prototype.

## References

1. S. Baron-Cohen, S. Wheelwright, J. Hill, Y. Raste, and I. Plumb, “The ‘Reading the Mind in the Eyes’ Test Revised Version,” *Journal of Child Psychology and Psychiatry*, 2001. [Cambridge Core](https://www.cambridge.org/core/journals/journal-of-child-psychology-and-psychiatry-and-allied-disciplines/article/reading-the-mind-in-the-eyes-test-revised-version-a-study-with-normal-adults-and-adults-with-asperger-syndrome-or-highfunctioning-autism/269621E672E0CD4CD20C33582130A8FB)
2. I. J. Goodfellow et al., “Challenges in Representation Learning: A Report on Three Machine Learning Contests,” 2013. [arXiv:1307.0414](https://arxiv.org/abs/1307.0414)
3. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
4. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
5. *Seeing Clearly* repository, final notebook, figures, and blog artifacts. [GitHub](https://github.com/Cece-AA/Seeing-Clearly)
