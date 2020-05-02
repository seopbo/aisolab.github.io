---
title: "Conditional BERT Contextual Augmentation"
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

> ***We propose a novel data augmentation method for labeled sentences called conditional BERT contextual augmentation.*** *Data augmentation methods are often applied to prevent overfitting and improve generalization of deep neural network models. Recently proposed contextual augmentation augments labeled sentences by randomly replacing words with more varied substitutions predicted by language model. BERT demonstrates that a deep bidirectional language model is more powerful than either an unidirectional language model or the shallow concatenation of a forward and backward model.* ***We retrofit BERT to conditional BERT by introducing a new conditional masked language model task. The well trained conditional BERT can be applied to enhance contextual augmentation.*** *Experiments on six various different text classification tasks show that our method can be easily applied to both convolutional or recurrent neural networks classifier to obtain obvious improvement.*

## 1. Introduction

- 기존의 data augmentation 방법론들은 데이터를 생성하여 Deep neural network-based model들이 overfit 또는 generalization을 잃는 것을 막았음. (e.g. image 데이터에 대한 random cropping, resizing 등)
- 그러나 text 데이터에 대해서 image 데이터에 쓰이는 data augmentation 기법들은 적용하기 힘듬.
  - semantic invariance나 label correctness를 보존할 수 없음.
- 기존의 text 데이터에 적용하는 data augmentation 기법들은 대략적으로 아래와 같음.
  - 특정 도메인에 결부된 handcraft rule 또는 pipeline -> loss of generality
  - synonym (유의어)에 기반한 replacement-based method -> 단어에 대한 유의어 자체가 굉장히 적으므로 다양한 패턴을 가진 sentence을 만들어 낼 수가 없음.
- replacement-based method 중 pretrained language model을 이용하여, synonym이 아닌 단어와 paradigmatic relation이 있는 단어로 replacement하는 방법 ([Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201))이 있지만 아래와 같은 한계점이 있음.
  - Shallow model을 사용 (bidirectional language model)
  - LSTM 사용 (transformer에 비해서 상대적으로 short range만 다룰 수 있음.)

 본 논문에서 제안하는 방법은 기존 연구 [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)를 BERT에 적용하는 **Conditional BERT contextual augmentation**을 제안함. 왜냐하면...

> *First, BERT is based on Transformer. Transformer provides us with a more structured memory for handling long-term dependencis in text. Second, BERT, as a deep bidirectional model, is strictly more powerful than the shallow concatenation of a left-to-right and right-to-left model.*

 또한 context에만 의존하는 masked language model로 masked token에 대하여 token을 replace할 경우, 기존의 label과 맞지않는 sentence로 바뀔 수도 있는 문제가 있음. 이러한 이유로 masked language model을 개선한 **conditional masked language model (C-MLM)**을 제안함.

> *The conditional masked language model randomly masks some of the tokens from an input, and objective is to predict a label-compatible word based on both its context and sentence label.*

## 2. Related Work

### 2.1 Fine-tuning on Pre-traine Language Model

- language mode을 pretraining하고 여러 downstream task에 fineutning하는 paradigm은 매우 효과적임.
- 본 논문에서는 masked language model을 이용, pretrained BERT로 generative task를 수행하고자 함.
- 특히 label compatibility를 만족하는 token replace를 하기위해서 C-MLM을 도입하고, 이를 이용해 task-specific data에 finetuning을 수행함.

### 2.2 Text Data Augmentation

- generation-based methods (e.g. GAN, VAE 등 활용)들은 continuous space (sentiment 또는 tense 등을 잘 encode하고 있는 latent space)로부터 sentence를 generate하려고 함.
  - label compatibility나 sentence readbility를 보장할 수 없음.
- 본 논문의 연구와 가장 비슷한 [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)의 경우
  - pretrained shallow language model을 이용하여, token을 replace
  - label compatibility를 만족하는 token replace를 하기위해서, conditilnal constraint (label 정보 주입)하는 방법을 도입함.

## 3. Conditional BERT Contextual Augmentation

### 3.1 Preliminary: Masked Language Model

#### 3.1.1 Bidirectional Language Model

- $N$ 개의 token으로 구성되는 sequence $S$, $<t_1, t_2, ..., t_N>$ 가 주어지면, bidirectional language model을 아래를 모델링함.
- 일반적으로 bidirectional language model은 각각 forward LM, backward LM이 따로 training이 되고, 특정 token을 encoding할 때, 두 LM의 결과를 concatenate함.

$$ p(t_1, t_2, ..., t_N) = \prod_{i=1}^{N} p(t_i \mid t_1,t_2,...,t_{i-1}) \tag{1} $$

$$ p(t_1, t_2, ..., t_N) = \prod_{i=1}^{N} p(t_i \mid t_{i+1},t_{i+2},...,t_{N}) \tag{2} $$

#### 3.1.2 Masked Language Model Task

- masked language model (MLM) task는 input tokens 중 일부를 랜덤하게 mask token `[MASK]`로 치환하고, 해당 token들의 context에 따라 원래 token이 무엇인 지 맞추는 task이다.
- MLM은 전체 input tokens가 아니라 일부의 token만 pretraining하는 데 활용하기 때문에, 꽤 많은 pretraining step이 필요하다.
- MLM task로 pretraining한 BERT로 text data augmentation을 할 수 있다.
  > *Pre-trained BERT can augment sentences through MLM task, by predicting new words in masked positions according to their context.*

### 3.2 Conditional BERT

conditional BERT의 architecture 자체는 BERT와 동일하지만, input representation을 구성하는 방법이 다르다.
- BERT의 경우: token embedding + segment embedding + positional embedding
- conditional BERT의 경우: token embedding + label embedding + positional embedding

![fig_1](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/fig_1.png)

#### 3.2.1 Conditional Masked Language Model

> *The conditional masked language model randomly masks some of tokens from the labeled sentence, and the objective is to predict the original vocabulary index of the masked word based on both its context and its label.*

- conditional MLM task를 수행하기위해서, pretrained BERT를 labeled dataset에 finetuning 해야함.
  - 해당 dataset이 label이 두 개인 경우, finetuning 시 segment embedding을 그대로 각 label에 할당하여 finetuning
  - 해당 dataset이 label이 두 개를 초과할 경우, label의 개수에 맞추어 label embedding을 초기화하고 finetuning
- conditional MLM task로 conditional BERT를 finetuning한 뒤, masked token에 대하여 label compatibility를 만족하는 token replace를 수행할 수 있음.

### 3.3 Conditional BERT Contextual Augmentation

![alg_1](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/alg_1.png)

## 4. Experiment

실험 설계는 선행 연구인 [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)에 맞춰서 수행함.
- pretrained BERT는 BERT BASE를 활용함.
- 아래와 같은 방식으로 실험을 수행
  1. dataset에 BERT로 MLM task를 수행하여 augment dataset 생성
  2. BERT를 conditional BERT로 dataset을 이용하여 finetuning 후, conditional MLM task를 수행하여 augment dataset 생성
  3. 위의 두 방법과 선행연구의 실험 결과와 비교

### 4.1 Datasets

선행 연구 [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)와 마찬가지로 table 1에 리스트업 된 데이터셋들을 활용함.

![table_1](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/table_1.png)

### 4.2 Text classification

#### 4.2.1 Sentence Classifier Structure

sentence classifier는 비교를 위해 [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)에서 사용했던 두 가지 모형을 활용함.
-  CNN 모형의 경우, [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)에서 제안한 architecture
-  RNN 모형의 경우, single layer LSTM + word embedding
- 두 모형 모두 training 시 early stopping 활용함.

#### 4.2.2 Hyper-parameters Setting

- 두 모형 모두 task-specitic dataset 별로 hyper-parameter들을 grid-search로 찾음.
- conditional BERT finetuning 시, epoch은 1 ~ 50, number of masked token은 1~2개로 설정

#### 4.2.3 Baselines

제안한 방법론과의 비교를 위해, 아래의 baseline을 제시
- w / synonym: WordNet 기반의 유의어 치환
- w / context: bidirectional language model
- w / context+label: bidirectional language model with label-constraint

#### 4.2.4 Experiments Results

conditional BERT로 task-specific dataset를 augment 한 뒤, 두 모형을 training 했을 때, 두 모형의 성능 증대가 다른 방법론에 비해서 개선됨을 확인함.

![table_2](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/table_2.png)

#### 4.2.5 Effect of Number of Fine-tuning Steps

conditional BERT를 적은 epoch으로 finetuning 한 뒤 dataset을 augment해도, 순수한 BERT MLM task로 dataset을 augment한 것보다 성능이 좋음을 확인함.

![table_3](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/table_3.png)

## 5. Connection to Style Transfer

Conditional MLM task를 이용하여, style transfer를 아래의 두 단계를 통해서 수행할 수도 있음을 확인함.
1. sentence에서 style (e.g. positive 또는 negative)과 관련된 token (e.g. postive 또는 negative에 가장 기여한 token)을 찾음.
2. 위 token을 masked하고, style을 기존과 반대로 부여한 뒤, condiontal MLM task를 수행함.

![table_4](https://raw.githubusercontent.com/aisolab/aisolab.github.io/master/_posts/_Conditional%20BERT%20Contextual%20Augmentation/table_4.png)

## 6. Conclusions and Future Work

> ***In this paper, we fine-tune BERT to conditional BERT by introducing a novel conditional MLM task. After being well trained, the conditional BERT can be applied to data augmentation for sentence classification tasks.*** *Experiment results show that our model outperforms several baseline methods obviously. Futhermore, we demonstrate that our conditional BERT can also be applied to style transfer task. In the future, (1) We will explore how to perform text data augmentation on imbalanced datasets with pre-trained language model, (2) we believe the idea of conditional BERT contextual augmentation is universal and will be applied to paragraph or document level data augmentation.*
