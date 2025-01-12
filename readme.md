# Using Rechours Framework to Evaluate the Effectiveness of ANS Model   **2024/12/31**
Augmented Negative Sampling (ANS)🚀 是一种用于协同过滤模型的增强负采样技术![new](/gif/new.gif)
本项目将ANS模型用于ReChorus框架进行测试😁，用于处理多种推荐算法的研究和复现工作😉

[郭怀泽的 GitHub Page](https://github.com/Zwt122544/ANS).<img src="/gif/github.gif" width="20" height="20">

[👉ANS的论文地址](https://arxiv.org/abs/2308.05972)

[👉ANS的github项目](https://github.com/Asa9aoTK/ANS-Recbole)
😀😀😀😀😀😀😀😀😀
## Requirement![new](/gif/new.gif)  

<details open>
<summary>点击展开安装依赖</summary>

克隆 repo，并要求在 [**Python>=3.8.0**]🌟 (https://www.python.org/)  环境中安装requirements.txt
<img src="/gif/python.gif" width="20" height="20">
```bash
pip install -r requirements.txt
```
其中库包含:
- torch==1.12.1
- cudatoolkit==10.2.89
- numpy==1.22.3
- ipython==8.10.0
- jupyter==1.0.0
- tqdm==4.66.1
- pandas==1.4.4
- scikit-learn==1.1.3
- scipy==1.7.3
- pickle
- yaml
</details>


 <img src="/gif/fcy1.png" width="2300" height="300">


## 模型结果展示

| Data                     | Metric                | AutoCF                 | LightGCN               | FPMC                   | SLRPlus                | GRU4Rec                | NeuMF                  |
|:-------------------------|:----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Grocery_and_Gourmet_Food | HR@5</br><br/>NDCG@5  | 0.1121</br><br/>0.0465 | 0.3858</br><br/>0.2659 | 0.3409</br><br/>0.2606 | 0.3242</br><br/>0.2249 | 0.3682</br><br/>0.2616 | 0.3261</br><br/>0.2242 |
| MIND_Large               | HR@5</br><br/>NDCG@5  | 0.2537</br><br/>0.0807 | 0.1078</br><br/>0.0631 | 0.1804</br><br/>0.1207 | 0.1098</br><br/>0.0716 | 0.2010</br><br/>0.1221 | 0.1020</br><br/>0.0638 |
| MovieLens-1M             | HR@5</br><br/>NDCG@5  | 0.6763</br><br/>0.2832 | 0.3520</br><br/>0.2382 | 0.4181</br><br/>0.2939 | 0.3693</br><br/>0.2455 | 0.4167</br><br/>0.2859 | 0.3319</br><br/>0.2277 |
 <img src="/gif/result.png">

## ANS模型结构
<img src="/gif/structure.png">

**🚀 点击展开查看代码**
<details open>
  
  
  ```python
sadasdasdas
 ```
</details>



## Usage
添加完数据集后
运行下面的命令：
```
python main.py --model_name ANS --emb_size 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression
```

- `--model_name`: 模型
- `--dataset`: 数据集名称
- `--path`: 数据集路径


## License

This project is licensed under the MIT License. It references ideas and methodologies from the following projects:

- **[👉ANS的论文地址](https://arxiv.org/abs/2308.05972)**
- **[👉ANS的github项目](https://github.com/Asa9aoTK/ANS-Recbole)**

