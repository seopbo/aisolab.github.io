---
title: "Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations"
excerpt: "Text data augmentation에 관한 논문"
toc: true
toc_sticky: true
categories:
  - Paper
tags:
  - Natural Language Processing
  - Data augmentation
---

## Abstract

> *We propose a novel data augmentation for labled sentences called* ***contextual augmentation.*** ***We assume an invariance that sentences are natural even if the words in the sentences are replaced with other words with paradigmatic relations. We stochastically replace words with other words that are predicted by a bi-directional language model at the word positions.*** Words predicted according to a context are numerous but appropriate for the augmentation of the original words. ***Furthermore, we retrofit a language model with a label-conditional architecture, which allows the model to augment sentences without breaking the label-compatibility.*** *Through the experiments for six various different text classification tasks, we demonstrate that the proposed method improves classifiers based on the convolutional or recurrent neural networks.*

## 1. Introduction

- machine learning 모델은 종종 training data에 overfitting -> generalization 저하
  - generalization은 training data의 quality와 size에 영향을 많이 받음.
- data augmentation은 trainign data의 size를 키워 machine learning 모델의 overfitting을 막아 generalization에 기여함.
  - data augmentation 방법들은 기본적으로 human knowledge에 근거함.
    > *e.g. "even if a picture is flipped, the class of an object should be unchanged".*
- 그러나 natural language에서는 image와 같이 다양한 domain에 쉽게 적용할 수 있는 data augmentation을 위한 rule을 찾기가 힘듦.
  - 특정 word의 synonym (e.g. WordNet)을 기반으로 word를 replace하여 data augmentation을 수행할 수 있지만, synonum 자체가 적어 많은 data를 생성할 수는 없음.
  - 다른 방법들은 domain에 종속되어 generality가 떨어짐.

본 논문에서는 **Contextual augmentation**이라는 방법을 제안하며, 특징은 아래와 같음.

- 특정 word의 synonym이 아니라 word의 paradigmatic relation에 기반하여 word를 replace하는 방법
  - 특정 word와 paradigmatic relation이 있는 word를 bidirectional language model을 이용하여 예측하고 sampling
- data augmentation을 수행 시, replace된 word에 따라서 원 sentence의 label과 생성된 sentence가 incompatible 할 가능성을 방지하는 방법
  - bidirectional language model로 특정 word와 pardigmatic relation 관계가 있는 word를 sampling 시, label을 condition으로 줌.

## 2. Proposed Method

word의 synonym에 기반한 data augmentation은 synonym 자체가 매우 적으므로, data augmentation으로 생성할 수 있는 sentence의 pattern이 매우 제한됨. 하지만 pardigmatic relation에 기반한 **Contextual augmentation**은 더 다양한 word로 sentence를 augmentation 할 수 있음.

### 2.1 Motivation

예를 들어, "the actors are fantisitic"이라는 sentence를 "actors"를 기준으로 augmentation을 할 때, synonym 기반인 지, paradigmatic relation 기반인 지에 따라 아래와 같은 차이가 있음.

- 특정 word의 synonym으로만 word를 replace해서 sentence를 augmentation할 경우
  - "actor"의 synonym인 "historion", "player", "thespian", "role player" 등 네 개의 sentence를 생성할 수 있음.
- 특정 word와 paradigmatic relation이 있는 word를 replace해서 sentence를 augmentation할 경우
  -  paradigmatic relation은 synonym을 포함하는 개념이므로 synonym 뿐만 아니라 non-synonym인 word도 augmentation에 이용할 수 있음. (e.g. "characters", "movies", "stories", "songs")

### 2.2 Word Prediction based on Context

![fig_1](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Contextual%20Augmentation_Data%20Augmentation%20by%20Words%20with%20Paradigmatic%20Relations/fig_1.png)

sentence를 word의 paradigmatic relation 관계가 있는 word로 replace하는 방법을 이용해 augmentation하기위해서는 pretrained language model이 필요함.

- pretrained language model을 이용해서 sentence $S$ 의 $i$ 번째 word $w_i$ 의 paradigmatic relation이 있는 word를 sampling

$$ p(\cdot \mid S \setminus \{w_i\})  $$

- temperature scaling을 도입하여, augmentation을 조절함. (contextual augmentation without a label-conditional architecture)
  - $\tau \rightarrow 0$ 이면 greedy sampling
  - $\tau \rightarrow \infty$ 이면 uniform sampling

$$ p_{\tau}(\cdot \mid S \setminus \{w_i\}) \propto p(\cdot \mid S \setminus \{w_i\})^ \frac{1}{\tau} $$

### 2.3 Conditional Constraint

paradigmatic relation만 고려하여 sentence를 augmentation을 하면, augmented sentencerk 원 sentence의 label과 상충될 가능성이 있음. (label-compatibiltiy issue)

- "the actors are fantastic"이라는 sentence를 "fantastic"의 paradigmatic relation이 있는 word로 replace할 경우
  - ("the actors are fantastic", positive) -> ("the actors are terrible", positive)

label-compatibiltiy를 보존하기위해서 최종적으로 label condition을 포함하는 **label-conditional LM**을 활용함. (contextual augmentation with a label-conditional architecture)

$$ p_{\tau}(\cdot \mid S \setminus \{w_i\}) \rightarrow p_{\tau}(\cdot \mid y,S \setminus \{w_i\})$$

## 3. Experiment

### 3.1 Settings

방법론의 검증을 위해서 LSTM-RNN, CNN 등을 classifier로 활용하여, 아래의 세 개의 data augmentation 방법론을 비교함. 

1. synonym-based augmentation
2. contextual augmentation without a label-conditional architecture
3. contextual augmentation with a label-conditional architecture

위의 (2), (3)의 contextual augmentation 방법을 위해서 아래의 작업이 필요함.

- contextual augmentation without a lbel-conditional architecture
  - bidirectional LSTM LM without the label-conditional architecture를 WikiText-103 corpus에 pretraining
- contextual augmentation with a label-conditional architecture
  - pretraining 후, 각각의 labeled dataset에 맞추어 bidirectional LSTM LM without the label-conditional architecture에 label-conditional architecture를 붙여 finetuning

### 3.2 Results

synonym-based 방법론보다 제안한 방법론이 더 효과적으로 작동함을 보임.

![table_1](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Contextual%20Augmentation_Data%20Augmentation%20by%20Words%20with%20Paradigmatic%20Relations/table_1.png)

![fig_2](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Contextual%20Augmentation_Data%20Augmentation%20by%20Words%20with%20Paradigmatic%20Relations/fig_2.png)

## 4. Related Work

> *We used a bidirectional LSTM LM which captures variable-length contexts with considering both the directions jointly. More importantly, we proposed a label-conditional architecture and demonstrated its effect both qualitatively and quantitatively.*

## 5. Conclusion

>***We proposed a novel data augmentation using numerous words given by a bi-directional LM, and further introduced a label-conditional architecture into the LM.*** *Experimentally, our method produced various words compatibly with the labels of original texts and improved neural classifiers more than the synonym-based augmentation. Our method is independent of any task-specific knowledge or rules, and can be generally and easily used for classification tasks in various domains.  
>On the other hand, the improvement by our method is sometimes marginal. Future work will explore comparison and combination with other generalization methods exploiting datasets deeply as well as our method.*