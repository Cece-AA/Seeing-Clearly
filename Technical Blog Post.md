# Seeing Clearly: Technical Blog Post


## Abstract

For many people, especially those on the autism spectrum, reading facial expressions can be stressful and uncertain. Our project, *Seeing Clearly*, was created to make that process a little easier by using a real-time webcam tool that identifies facial expressions and translates them into emotional categories. We hypothesized that fine-tuning a ResNet-18 model on the FER-2013 dataset would allow the system to reach at least 70% test accuracy and be useful as the basis for an assistive interface. After training and testing the model, our selected transfer-learning model reached 68.56% accuracy on the held-out FER-2013 test set and 69.55% validation accuracy during model selection, falling just below our original goal. However, this project also showed that accuracy alone does not fully capture whether a tool like this would be helpful in real social situations. Some expression categories were predicted much more reliably than others, meaning that confidence scores and uncertainty thresholds are important for making the tool more trustworthy. Overall, *Seeing Clearly* demonstrates both the promise and the limits of using facial-expression recognition as a form of real-time social support.


## 1. Introduction

Facial expressions are a major part of how people read social situations. In many conversations, people use facial cues automatically to judge tone, emotion, and reaction. However, for some users, especially people who have difficulty interpreting facial expressions, those cues can feel unclear, overwhelming, or easy to misread. *Seeing Clearly* was designed as an assistive facial-expression recognition tool that could make these cues easier to notice in real time.

The goal of this project is not to claim that a model can fully understand what someone is feeling. Facial expressions do not always match a person’s internal emotional state, and treating them as definite proof of emotion would be irresponsible. Instead, the purpose of *Seeing Clearly* is to provide cautious visual support. The system should offer possible cues, not final judgments.

This distinction matters because a standard classifier is built to always choose one label, even when it is unsure. In a real social setting, that can be risky. If the model is uncertain, the expression is ambiguous, or the prediction is unstable across frames, the interface should not present the result as if it is completely reliable. Instead, it can soften the output, wait for a clearer signal, or avoid showing a prediction at all.

This makes the central question of the project: **When is a transfer-learned facial-expression classifier reliable enough to show a cue, and when should the interface withhold or soften its output?**


## 2. Hypothesis

The original hypothesis was that a fine-tuned ResNet-18 model using transfer learning would reach at least **70% accuracy** on the FER-2013 dataset while still being lightweight enough to run in a real-time webcam application.

For the final project, this hypothesis can be divided into two parts:

1. **Performance hypothesis:** A transfer-learned ResNet-18 trained on FER-2013 will approach 70% accuracy on a held-out test set and perform far above random chance.

2. **Interface hypothesis:** For an assistive tool, showing uncertainty through confidence thresholds and temporal smoothing will be more useful and responsible than always displaying the model’s top predicted class.

The final results partially support both hypotheses. The selected model came close to the 70% benchmark, but did not fully meet it. However, the uncertainty analysis strongly supports the second hypothesis. The model’s confidence scores contained useful information, and some expression categories were predicted much more reliably than others. This suggests that an assistive interface should not treat every prediction equally. A system like *Seeing Clearly* is safer and more useful when it can hedge, wait, or stay quiet instead of forcing a label onto every frame.


## 3. Literature Review

This project connects research from psychology with work in machine learning. Baron-Cohen et al.’s [1] revised “Reading the Mind in the Eyes” test shows that people differ in how easily they interpret mental states from facial cues. This is especially relevant for autism research, where reading facial expressions and other social cues can be difficult or stressful. However, this does not mean that a person’s emotions can be perfectly read from their face. For *Seeing Clearly*, we emphasize that facial cues can be helpful in social situations, but they should be treated as possible clues, not definite proof of what someone is feeling.

The project uses the FER-2013 dataset, which was introduced by Goodfellow et al [2] through the ICML 2013 representation learning challenge. FER-2013 is commonly used because it gives a clear benchmark with more than 35,000 grayscale face images labeled into seven categories: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. At the same time, it is a limited dataset. The images are only 48 by 48 pixels, the labels are broad, and the dataset is imbalanced. Some expressions are also much harder to separate than others. Negative emotions like fear, sadness, anger, and disgust can be more subtle and visually similar, while happiness and surprise are usually more visually obvious. Because of this, doing well on FER-2013 does not automatically mean the model would work well in a real social setting.

For the model, we used ResNet-18 which came from the residual-learning framework introduced by He et al [3]. ResNet models use skip connections which make it easier to train deeper neural networks. ResNet-18 was a good fit for this project because it is strong without being too heavy. It can use features learned from ImageNet pretraining, but it is still lightweight enough to use in a real-time webcam prototype.

This project also draws from Guo et al.’s [4] work on calibration in neural networks. Their work shows that even models with high accuracy can still be poorly calibrated, meaning the model’s confidence score does not always match how likely it is to be correct. This matters a lot for an assistive tool. In real social situations, a model that is confidently wrong could be more harmful than a model that is less forceful about its predictions. Because of that, *Seeing Clearly* uses confidence scores as part of the design.

Overall, the literature supports the main idea behind this project. Facial cues can matter, but they should be handled carefully. FER-2013 gives us a useful benchmark, but it has its limits. ResNet-18 gives us a practical model for real-time use, and calibration research shows why the interface should not treat every prediction the same. 


## 4. Methods

### Preprocessing

FER-2013, which contains grayscale face crops labeled as `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, or `surprise`. Because ResNet-18 expects three-channel images at higher spatial resolution than 48x48, the preprocessing pipeline converts grayscale inputs into three channels and resizes them to 224x224. During training, the notebook applies augmentation with random crops, flips, rotations, affine transforms, contrast adjustments, and random erasing. These choices were made for the low-resolution dataset where overfitting is a serious risk.

The input *x* is a preprocessed facial image from FER-2013, and the output *y* is one of seven expression labels: angry, disgust, fear, happy, neutral, sad, or surprise. The model is trained as a seven-class classification problem using cross-entropy loss. Performance is evaluated using held-out accuracy, validation accuracy, per-class accuracy, confusion matrices, confidence-threshold tradeoffs, and expected calibration error.

![Figure 1. FER-2013 Classes.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/fer2013_examples.png)

**Figure 1.** One FER-2013 training example from each class. 

### Model Architecture

The classifier uses ResNet-18 with a modified head. The final fully connected layer is replaced by a dropout layer followed by a seven-class linear classifier. This is a standard and effective transfer-learning pattern. It keeps a strong pretrained feature extractor, then adapts the final decision layers to the target task. In the live app, the model is paired with OpenCV Haar-cascade face detection and applies CLAHE to normalize contrast on detected face regions.

### Experimental design

Some earlier continuation experiments were informative but not ideal as final evidence because they involved checkpoints with more complicated histories. The final comparison instead fixes a single stratified train/validation split and evaluates all candidate models under that shared setup.

The main candidates were:

| Candidate | Best epoch | Validation accuracy | Macro accuracy | Weak-class accuracy | Selection score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean transfer baseline | 11 | 69.55% | 66.41% | 57.05% | 67.98% |
| Balanced, weak boost 1.2 | 14 | 68.69% | 67.61% | 57.94% | 66.11% |
| Balanced, weak boost 1.4 | 14 | 67.72% | 66.78% | 58.17% | 65.44% |

**Table 1.** Final clean-model comparison from the repository’s selected experiment metadata. The balanced variants helped weaker classes slightly, but the plain transfer baseline won the final selection criterion.

This table shows that class-balanced strategies did improve some weak classes, especially those that are harder and more socially consequential, but they did not produce the best overall clean selection score. Thus, surprisingly, the final model is not the most complicated one. Rather, it is the simplest candidate that held up best under the protocol.

### Prototype interface

In the live prototype predictions are smoothed across recent frames using a short temporal average which helps reduce label flicker. The interface then turns predicted expressions into prompts such as “Ask if everything is okay”. Additionally, the UI design encodes uncertainty and social caution into the prototype itself.


## 5. Results

The selected final model reaches **68.56% accuracy on the FER-2013 held-out test set**. That result falls just short of our original 70% benchmark hypothesis, but it still represents a strong improvement over early project runs and a substantial gain over random chance. It supports the claim that transfer learning with ResNet-18 is a viable baseline for this task.

![Figure 2. Training history for the selected model.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/training_history.png)

**Figure 2.** Training history for the selected clean model. The training curves suggest a reasonably healthy fit rather than catastrophic overfitting. Accuracy improved steadily, and the selected epoch is chosen on validation behavior rather than simply the last epoch. Results were also broken down by class.

![Figure 3. Row-normalized confusion matrix.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/confusion_matrix_normalized.png)

**Figure 3.** Row-normalized confusion matrix on the held-out FER-2013 test set.

The confusion matrix shows that performance is uneven across different classes. Classes such as `happy` and `surprise` are easier for the model, while classes such as `fear` and `sad` remain harder. This makes sense because negative-expressions are more subtle, and thus harder to detect, than visibly expressive emotions like happiness and surprise.

![Figure 4. Per-class accuracy.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/per_class_accuracy.png)

**Figure 4.** Per-class accuracy emphasizes that the model is not uniformly reliable across expression categories.

Instead of assuming that every argmax prediction should be shown, the project measures how accuracy changes as the confidence threshold rises. The result is an interpretable coverage vs. reliability tradeoff.

![Figure 5. Confidence threshold tradeoff.](https://raw.githubusercontent.com/Cece-AA/Seeing-Clearly/main/assets/figures/confidence_threshold_tradeoff.png)

**Figure 5.** Raising the confidence threshold reduces coverage but improves the reliability of the predictions that remain. This figure strongly supports the second hypothesis that an assistive interface does not need to label every frame. If it only surfaces predictions above a confidence threshold, it can become more trustworthy at the cost of covering fewer situations.

The repository also includes a reliability diagram and reports an expected calibration error of about 0.067 for the selected model. This suggests that confidence scores are informative enough to use as part of the interface logic. Confidence gives meaningful signals about when the model should hedge.


## 6. Limitations

The project has several important limitations. First, the FER-2013 labels should not be treated as true emotions in a deep psychological sense. They are broad dataset categories assigned to still images. This means the model is learning to match cropped face images to benchmark labels, not to understand what someone is actually feeling. In a real social situation, a person’s facial expression may not fully reflect their internal emotional state, so the system has to be framed as a source of possible cues rather than emotional truth.

A second limitation is the quality of the dataset itself. FER-2013 is low-resolution, visually noisy, and imbalanced. These issues make it a useful challenge dataset, but they also limit how much we can generalize from the results. A model that performs well on FER-2013 may still struggle with different lighting conditions, ages, cultures, camera qualities, and personal expression styles.

The model also had more difficulty with some expression categories than others. This is especially important for negative emotions, which can be more subtle and visually overlapping. Expressions like fear, sadness, anger, and disgust may share similar facial features or appear less exaggerated than a clear smile, making them harder for the model to separate. This matters because misclassifying negative expressions could be especially misleading in an assistive setting. For that reason, the system should avoid presenting these predictions too confidently and should rely on confidence thresholds or softened language when the model is uncertain.


## 7. Ethics and Responsible Use

There are clear ethical issues that must be addressed. First, facial-expression classification risks overstating what is observable from a face. Expressions are shaped by culture, personality, among other factors. An assistive tool that presents outputs as facts could easily encourage overconfidence in weak inferences. The project tries to mitigate this by describing predictions as “soft cues” and pairing them with prompts rather than blunt declarations.

Second, the model may not perform equally well across demographic groups. The dataset does not contain the demographic annotations needed for a meaningful subgroup analysis. That means the model could perform unevenly across race, age, gender, disability, or other axes without the current evaluation detecting it. This can become problematic in a real live-application.

For those reasons, *Seeing Clearly* should be used with caution in real-world settings. For now, the human user's judgement, combined with the uncertainty benchmarks, should take precedent and the prototype should be seen as an assistive, suggestive tool rather than a replacement for human judgmen

## 8. Discussion and Conclusion

The final ResNet-18 model reached **68.56% accuracy** on the held-out FER-2013 test set, falling slightly below the original **70% accuracy** benchmark. This partially supports the performance hypothesis: transfer learning produced a usable facial-expression classifier for a real-time prototype, but the model did not fully meet the target set at the beginning of the project.

The results also show that overall accuracy is not enough to evaluate an assistive facial-expression recognition system. The model performed unevenly across expression categories. More visually distinct expressions, such as happiness and surprise, were easier to classify, while negative expressions such as fear, sadness, anger, and disgust were more difficult. These categories are often more subtle and visually overlapping, which makes them harder for the model to separate. This matters because errors in these classes could be especially misleading in a social-support setting.

The confidence-threshold analysis supports the interface hypothesis more strongly. When the system withholds low-confidence predictions, the predictions that remain become more reliable. This suggests that the interface should not always display the top predicted class. Instead, it should use confidence thresholds, softened language, and temporal smoothing to reduce unstable or misleading outputs.

Overall, *Seeing Clearly* shows that transfer learning can support a workable FER-2013 webcam prototype, but it also shows the limits of treating facial-expression recognition as a simple classification problem. The model should not be understood as reading emotions. It is better framed as a cautious assistive tool that provides possible cues when the prediction is confident enough.

Future work should focus on temperature scaling, a quantitative evaluation of temporal smoothing, testing on more diverse datasets, and user studies that examine whether the prompts are actually helpful. These steps would make the system more reliable and move the project closer to a responsible assistive research prototype.



## References

1. S. Baron-Cohen, S. Wheelwright, J. Hill, Y. Raste, and I. Plumb, “The ‘Reading the Mind in the Eyes’ Test Revised Version,” *Journal of Child Psychology and Psychiatry*, 2001. [Cambridge Core](https://www.cambridge.org/core/journals/journal-of-child-psychology-and-psychiatry-and-allied-disciplines/article/reading-the-mind-in-the-eyes-test-revised-version-a-study-with-normal-adults-and-adults-with-asperger-syndrome-or-highfunctioning-autism/269621E672E0CD4CD20C33582130A8FB)
2. I. J. Goodfellow et al., “Challenges in Representation Learning: A Report on Three Machine Learning Contests,” 2013. [arXiv:1307.0414](https://arxiv.org/abs/1307.0414)
3. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
4. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
5. *Seeing Clearly* repository, final notebook, figures, and blog artifacts. [GitHub](https://github.com/Cece-AA/Seeing-Clearly)
