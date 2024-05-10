---
layout: about
title:
permalink: /
# subtitle: Research Assistant @ AutonLab, CMU | BSc in AI @ Poznan University of Technology

# profile:
  # align: right
  # image: prof_pic.jpg
  # image_circular: false # crops the image to make it circular
  # more_info: >
  #   <p>555 your office number</p>
  #   <p>123 your address street</p>
  #   <p>Your City, State 12345</p>

news: False # includes a list of news items
latest_posts: False # includes a list of the newest posts
selected_papers: True # includes a list of papers marked as "selected={true}"
selected_projects: True # includes a list of projects marked as "selected={true}"
social: true # includes social icons at the bottom of the page
---

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 34px; font-weight: bold; margin-bottom: 22px;">
      MOMENT: A FAMILY OF OPEN TIME-SERIES FOUNDATION MODELS</h1>
    <p style="font-size: 18px; line-height: 1.0;">
      Mononito Goswami<sup>1</sup> Konrad Szafer<sup>*1</sup> Arjun Choudhry<sup>*1</sup> Yifu Cai<sup>1</sup> Shuo Li<sup>2</sup> Artur Dubrawski<sup>1</sup>
    </p>
    <p style="font-size: 16px; line-height: 1.0;">
      <sup>1</sup>Carnegie Mellon University, <sup>2</sup>University of Pennsylvania
    </p>
    <p style="font-size: 12px; font-weight: 120; line-height: 0.2; font-style: italic; color: #505050;">
      * Equal contribution
    </p>
    <p style="font-size: 20px; line-height: 1.0;">2024</p>
</div>



---

We introduce MOMENT, a family of open-source foundation models for general-purpose time-series
analysis. Pre-training large models on time-series data is challenging due to (1) the absence of a
large and cohesive public time-series repository, and (2) diverse time-series characteristics which
make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these
models, especially in scenarios with limited resources, time, and supervision, are still in their
nascent stages. To address these challenges, we compile a large and diverse collection of public
time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to
unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark
to evaluate time-series foundation models on diverse tasks and datasets in limited supervision
settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models
with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical
observations about large pre-trained time-series models.

<div style="text-align: center;">
    <a href="https://arxiv.org/abs/2402.03885" style="background-color: #EEEEEE; color: black; padding: 10px 20px; text-decoration: none; display: inline-block; margin: 4px 2px; cursor: pointer; border-radius: 12px;">arXiv</a>
    <a href="https://github.com/moment-timeseries-foundation-model/moment" style="background-color: #EEEEEE; color: black; padding: 10px 20px; text-decoration: none; display: inline-block; margin: 4px 2px; cursor: pointer; border-radius: 12px;">GitHub</a>
    <a href="https://huggingface.co/AutonLab/MOMENT-1-large" style="background-color: #EEEEEE; color: black; padding: 10px 20px; text-decoration: none; display: inline-block; margin: 4px 2px; cursor: pointer; border-radius: 12px;">Model Weights</a>
    <a href="https://huggingface.co/datasets/AutonLab/Timeseries-PILE" style="background-color: #EEEEEE; color: black; padding: 10px 20px; text-decoration: none; display: inline-block; margin: 4px 2px; cursor: pointer; border-radius: 12px;">Time-series PILE</a>
</div>

---

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 30px; font-weight: bold; margin-bottom: 22px;">
      The Time-series Pile</h1>
</div>

<!-- Limiting factor for pre-training large time-series models from scratch was the lack of a large cohesive public time-series data repositories. Therefore, we compiled The Time-series Pile, a large collection of publicly available data from diverse domains, ranging from healthcare to engineering to finance. The Time-series Pile comprises of over 5 public time-series databases, from several diverse domains for pre-training and evaluation

<p align="center">
  <img src="assets/img/PILE-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      An overview of the datasets and their parameters comprising the Time-series PILE. </h1>
</div> -->

We compiled a large collection of publicly available datasets from diverse domains into the Time Series Pile. It has 13 unique domains of data, which includes 20.085 GB worth of 13M unique time series and 1.23 billion timestamps (including channels). The data has been collated from more than 5 task-specific, widely-used public repositories resulting in a large number of time series spanning diverse domains, and time series characteristics such as lengths, amplitudes, and temporal resolutions. Some details about these public repositories are as follows:

- **Informer long-horizon forecasting datasets** ([Zhou et al., 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17325)) is a collection of 9 datasets that are widely used to evaluate long-horizon forecasting performance: 2 hourly and minutely subsets of the [Electricity Transformer Temperature (ETT)](https://ojs.aaai.org/index.php/AAAI/article/view/17325), [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), [Traffic](http://pems.dot.ca.gov/), [Weather](https://www.bgc-jena.mpg.de/wetter/), [Influenza-like Illness (ILI)](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html), and [Exchange-rate](https://arxiv.org/abs/1703.07015).

- **Monash time series forecasting archive** ([Godahewa et al., 2021)](https://openreview.net/forum?id=wEc1mgAjU-)) is a collection of 58 publicly available short-horizon forecasting datasets with a total of over 100K time series, spanning a variety of domains and temporal resolutions.

- **UCR/UEA classification archive** ([Dau et al., 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)) comprises of 159 time series datasets which are frequently used to benchmark classification algorithms. These datasets belong to seven different categories (Image Outline, Sensor Readings, Motion Capture, Spectrographs, ECG, Electric Devices, and Simulated Data), and vary substantially in terms of the number of classes and the size of the training set.

- **TSB-UAD anomaly benchmark** ([Paparrizos et al., 2022b](https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf)) is a recent collection of 1980 univariate time series with labeled anomalies from 18 anomaly detection datasets proposed over the past decade. This collection includes both synthetic and real-world time series originating from a wide range of sources such as the human body, spaceships, environment, and web serves.

---

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 30px; font-weight: bold; margin-bottom: 22px;">
      General Time-series Pre-training</h1>
</div>

Unlike text and images, which have largely consistent sampling rates and number of channels, time-series frequently vary in their temporal resolution, number of channels, lengths, and amplitudes, and sometimes have missing values. As a result, large-scale mixed dataset pre-training is largely unexplored. Instead, most methods are trained on a single dataset, and transferred across multiple datasets, but with modest success.

<p align="center">
  <img src="assets/img/MOMENT.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Overview of MOMENT pre-training procedure and architecture. </h1>
</div>

---

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 30px; font-weight: bold; margin-bottom: 22px;">
      Holistic Multi-task Evaluation</h1>
</div>

Comprehensive benchmarks to evaluate time-series foundation models on diverse datasets and tasks are in their nascent stages. To evaluate MOMENT, we build on the multi-task time-series modeling benchmark first proposed by [Wu et al. [2023]](https://arxiv.org/abs/2210.02186) along multiple dimensions. For each of the **5** time-series modeling tasks, namely, short- and long-horizon forecasting, classification, anomaly detection, and imputation we evaluate MOMENT against (1) both state-of-the-art deep learning as well as statistical baselines, on (2) more task-specific datasets, (3) using multiple evaluation metrics, (4) exclusively in limited supervision settings (e.g., zero-shot imputation, linear probing for forecasting, unsupervised representation learning for classification). 

<!-- LONG HORIZON FORECASTING -->

<p align="center">
  <img src="assets/img/Eval-Long-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Long-term forecasting performance measured using Mean Squared Error (MSE) and Mean Absolute Error (MAE). </h1>
</div>

<!-- SHORT HORIZON FORECASTING -->

<p align="center">
  <img src="assets/img/Eval-Short-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Zero-shot short-horizon forecasting performance on a subset of the M3 and M4 datasets measured using sMAPE. </h1>
</div>

<!-- CLASSIFICATION -->

<p align="center">
  <img src="assets/img/Eval-Classification-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Classification accuracy of methods across 91 UCR datasets. Methods with mean and median accuracy higher than MOMENT are in **bold**. </h1>
</div>

<!-- IMPUTATION -->

<p align="center">
  <img src="assets/img/Eval-Imputation-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Imputation Results. MOMENT with linear probing achieved the lowest reconstruction error on all ETT datasets. </h1>
</div>

<!-- IMPUTATION -->

<p align="center">
  <img src="assets/img/Eval-Anomaly-table.png" alt="Description of image" style="max-width: 100%; height: auto;">
</p>

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 12px; margin-top: -10px; margin-bottom: 22px;">
      Anomaly detection performance averaged over 44 time-series from the UCR Anomaly Archive. </h1>
</div>

---

<div style="text-align: center; font-family: Arial, sans-serif;">
    <h1 style="font-size: 24px; margin-top: 14px; margin-bottom: 18px;">
      BibTeX </h1>
</div>

```bibtex
@inproceedings{goswami2024moment,
  title={MOMENT: A Family of Open Time-series Foundation Models},
  author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
  booktitle={International Conference on Machine Learning},
  year={2024},
  abstract={We introduce MOMENT, a family of open-source foundation models for general-purpose time-series analysis. Pre-training large models on time-series data is challenging due to (1) the absence of a large and cohesive public time-series repository, and (2) diverse time-series characteristics which make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these models, especially in scenarios with limited resources, time, and supervision, are still in their nascent stages. To address these challenges, we compile a large and diverse collection of public time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark to evaluate time-series foundation models on diverse tasks and datasets in limited supervision settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical observations about large pre-trained time-series models. Our code is available anonymously at anonymous.4open.science/r/BETT-773F/.}
}
```
